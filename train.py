import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:
    def __init__(self, data_dict):
        self.train_data_dir = data_dict['train_data_dir']
        print("train data:")
        print_tiff_filenames(self.train_data_dir)

        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.disp_freq = data_dict['disp_freq']
        self.train_continue = data_dict['train_continue']

        self.hyperparameters = data_dict['hyperparameters']

        self.model_name = self.hyperparameters['model_name']
        self.UNet_base = self.hyperparameters['UNet_base']
        self.num_epoch = self.hyperparameters['num_epoch']
        self.batch_size = self.hyperparameters['batch_size']
        self.lr = self.hyperparameters['lr']
        self.patience = self.hyperparameters.get('patience', 10)  # Load patience with a default value

        self.device = get_device()

        # Create result and checkpoint directories with new naming convention
        self.results_dir, self.checkpoints_dir = create_result_dir(
            self.project_dir, self.project_name, self.hyperparameters, self.train_data_dir)
        self.train_results_dir = create_train_dir(self.results_dir)

        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')

    def save(self, checkpoints_dir, model, optimizer, epoch, best_train_loss):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_train_loss': best_train_loss,
            'hyperparameters': self.hyperparameters
        }, os.path.join(checkpoints_dir, 'best_model.pth'))

    def load(self, checkpoints_dir, model, device, optimizer):
        checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        dict_net = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(dict_net['model'])
        optimizer.load_state_dict(dict_net['optimizer'])
        epoch = dict_net['epoch']
        best_train_loss = dict_net.get('best_train_loss', float('inf'))
        self.hyperparameters = dict_net.get('hyperparameters', self.hyperparameters)

        print(f'Loaded {epoch}th network with hyperparameters: {self.hyperparameters}, best train loss: {best_train_loss:.4f}')

        return model, optimizer, epoch, best_train_loss

    def get_model(self):
        if self.model_name == 'UNet3':
            return UNet3(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet4':
            return UNet4(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet5':
            return UNet5(base=self.UNet_base).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train(self):
        start_time = time.time()
        mean, std = compute_global_mean_and_std(self.train_data_dir, self.checkpoints_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        transform_train = transforms.Compose([
            Normalize(mean, std),
            RandomCrop(output_size=(64,64)),
            RandomHorizontalFlip(),
            ToTensor()
        ])

        transform_inv_train = transforms.Compose([
            BackTo01Range(),
            ToNumpy()
        ])

        crop_tiff_depth_to_divisible(self.train_data_dir, self.batch_size)

        ### make dataset and loader ###
        dataset_train = TwoSliceDataset(self.train_data_dir, transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        ### initialize network ###
        model = self.get_model()
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        st_epoch = 0
        best_train_loss = float('inf')
        patience_counter = 0  # Initialize patience counter

        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            model, optimizer, st_epoch, best_train_loss = self.load(self.checkpoints_dir, model, self.device, optimizer)
            model = model.to(self.device)

        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            model.train()  # Ensure model is in training mode
            train_loss = 0.0

            for batch, data in enumerate(loader_train, 1):
                optimizer.zero_grad()
                input_slice, target_img = [x.squeeze(0).to(self.device) for x in data]
                output_img = model(input_slice)

                loss = criterion(output_img, target_img)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % self.disp_freq == 0:
                # Assuming transform_inv_train can handle the entire stack
                input_img = transform_inv_train(input_slice)[..., 0]
                target_img = transform_inv_train(target_img)[..., 0]
                output_img = transform_inv_train(output_img)[..., 0]

                for j in range(target_img.shape[0]):
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_input.png"), input_img[j, :, :], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_target.png"), target_img[j, :, :], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_output.png"), output_img[j, :, :], cmap='gray')

            avg_train_loss = train_loss / len(loader_train)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

            print(f'Epoch [{epoch}/{self.num_epoch}], Train Loss: {avg_train_loss:.4f}')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                self.save(self.checkpoints_dir, model, optimizer, epoch, best_train_loss)
                patience_counter = 0  # Reset patience counter
                print(f"Saved best model at epoch {epoch} with training loss {best_train_loss:.4f}.")
            else:
                patience_counter += 1  # Increment patience counter
                print(f'Patience Counter: {patience_counter}/{self.patience}')

            # Check for early stopping
            if patience_counter >= self.patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break

        self.writer.close()

