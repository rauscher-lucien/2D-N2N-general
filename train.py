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
        # Initialize the trainer with the provided data and hyperparameters
        self.train_data_dir = data_dict['train_data_dir']
        print("train data:")
        print_tiff_filenames(self.train_data_dir)  # Print available training data

        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.disp_freq = data_dict['disp_freq']  # How frequently (in epochs) to display results
        self.train_continue = data_dict['train_continue']  # Whether to continue training from a checkpoint

        # Load hyperparameters (model architecture, number of epochs, batch size, learning rate, etc.)
        self.hyperparameters = data_dict['hyperparameters']

        self.model_name = self.hyperparameters['model_name']
        self.UNet_base = self.hyperparameters['UNet_base']
        self.num_epoch = self.hyperparameters['num_epoch']
        self.batch_size = self.hyperparameters['batch_size']
        self.lr = self.hyperparameters['lr']
        self.patience = self.hyperparameters.get('patience', 10)  # Load patience with a default value of 10

        # Determine the device (CPU or GPU) to run the model
        self.device = get_device()

        # Create directories to store results and model checkpoints
        self.results_dir, self.checkpoints_dir = create_result_dir(
            self.project_dir, self.project_name, self.hyperparameters, self.train_data_dir)
        self.train_results_dir = create_train_dir(self.results_dir)

        # TensorBoard logging for visualizing the training process
        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')

    def save(self, checkpoints_dir, model, optimizer, epoch, best_train_loss):
        # Save the current state of the model, optimizer, and training information
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({
            'model': model.state_dict(),  # Save model weights
            'optimizer': optimizer.state_dict(),  # Save optimizer state
            'epoch': epoch,  # Save the current epoch
            'best_train_loss': best_train_loss,  # Save the best training loss so far
            'hyperparameters': self.hyperparameters  # Save hyperparameters for reproducibility
        }, os.path.join(checkpoints_dir, 'best_model.pth'))

    def load(self, checkpoints_dir, model, device, optimizer):
        # Load the saved model checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        # Load model, optimizer, and training details from the checkpoint
        dict_net = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(dict_net['model'])  # Load model state
        optimizer.load_state_dict(dict_net['optimizer'])  # Load optimizer state
        epoch = dict_net['epoch']  # Load the saved epoch
        best_train_loss = dict_net.get('best_train_loss', float('inf'))  # Load the best training loss
        self.hyperparameters = dict_net.get('hyperparameters', self.hyperparameters)  # Load hyperparameters

        print(f'Loaded {epoch}th network with hyperparameters: {self.hyperparameters}, best train loss: {best_train_loss:.4f}')

        return model, optimizer, epoch, best_train_loss

    def get_model(self):
        # Return the appropriate UNet model based on the configuration
        if self.model_name == 'UNet3':
            return UNet3(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet4':
            return UNet4(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet5':
            return UNet5(base=self.UNet_base).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train(self):
        # Start the training process
        start_time = time.time()

        # Compute and save global mean and standard deviation for normalization
        mean, std = compute_global_mean_and_std(self.train_data_dir, self.checkpoints_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        # Define data transformations for training (normalization, cropping, flipping, tensor conversion)
        transform_train = transforms.Compose([
            Normalize(mean, std),
            RandomCrop(output_size=(64,64)),
            RandomHorizontalFlip(),
            ToTensor()
        ])

        # Define inverse transformation to visualize the training results
        transform_inv_train = transforms.Compose([
            BackTo01Range(),
            ToNumpy()
        ])

        # Adjust the depth of the TIFF stacks to be divisible by batch size
        crop_tiff_depth_to_divisible(self.train_data_dir, self.batch_size)

        ### Create dataset and data loader ###
        dataset_train = TwoSliceDataset(self.train_data_dir, transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        ### Initialize the model, loss function, and optimizer ###
        model = self.get_model()  # Get the appropriate UNet model
        criterion = nn.MSELoss().to(self.device)  # Mean Squared Error loss function
        optimizer = torch.optim.Adam(model.parameters(), self.lr)  # Adam optimizer

        st_epoch = 0  # Starting epoch
        best_train_loss = float('inf')  # Track the best training loss
        patience_counter = 0  # Initialize patience counter for early stopping

        # If continuing from a previous checkpoint, load the model and optimizer states
        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            model, optimizer, st_epoch, best_train_loss = self.load(self.checkpoints_dir, model, self.device, optimizer)
            model = model.to(self.device)

        # Training loop over the number of epochs
        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            model.train()  # Set the model to training mode
            train_loss = 0.0

            for batch, data in enumerate(loader_train, 1):
                optimizer.zero_grad()  # Clear gradients
                input_slice, target_img = [x.squeeze(0).to(self.device) for x in data]  # Load input and target to device
                output_img = model(input_slice)  # Forward pass

                # Compute loss
                loss = criterion(output_img, target_img)
                train_loss += loss.item()
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            # Display results at specified frequency
            if epoch % self.disp_freq == 0:
                input_img = transform_inv_train(input_slice)[..., 0]
                target_img = transform_inv_train(target_img)[..., 0]
                output_img = transform_inv_train(output_img)[..., 0]

                # Save images of input, target, and output to the results directory
                for j in range(target_img.shape[0]):
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_input.png"), input_img[j, :, :], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_target.png"), target_img[j, :, :], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_output.png"), output_img[j, :, :], cmap='gray')

            # Compute average training loss for the epoch
            avg_train_loss = train_loss / len(loader_train)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)  # Log loss to TensorBoard

            print(f'Epoch [{epoch}/{self.num_epoch}], Train Loss: {avg_train_loss:.4f}')

            # Save the model if it has the best training loss so far
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                self.save(self.checkpoints_dir, model, optimizer, epoch, best_train_loss)
                patience_counter = 0  # Reset patience counter
                print(f"Saved best model at epoch {epoch} with training loss {best_train_loss:.4f}.")
            else:
                patience_counter += 1  # Increment patience counter if no improvement
                print(f'Patience Counter: {patience_counter}/{self.patience}')

            # Stop training early if patience is exceeded (early stopping)
            if patience_counter >= self.patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break

        self.writer.close()  # Close TensorBoard writer after training is finished


