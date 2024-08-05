import os
import sys
import argparse
import logging
import glob
import torch
import numpy as np
import tifffile
from torchvision import transforms

sys.path.append(os.path.join(".."))

from model import *
from transforms import *
from utils import *
from dataset import *

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_file='logging.log'):
    logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def load_hyperparameters(checkpoints_dir, device='cpu'):
    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    dict_net = torch.load(checkpoint_path, map_location=device)
    hyperparameters = dict_net['hyperparameters']
    epoch = dict_net['epoch']

    return hyperparameters, epoch

def load_model(checkpoints_dir, model, optimizer=None, device='cpu'):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  

    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
    dict_net = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(dict_net['model'])
    optimizer.load_state_dict(dict_net['optimizer'])
    epoch = dict_net['epoch']

    model.to(device)

    print(f'Loaded {epoch}th network with hyperparameters: {dict_net["hyperparameters"]}')

    return model, optimizer, epoch

def get_model(model_name, UNet_base):
    if model_name == 'UNet3':
        return UNet3(base=UNet_base)
    elif model_name == 'UNet4':
        return UNet4(base=UNet_base)
    elif model_name == 'UNet5':
        return UNet5(base=UNet_base)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Process inference parameters.')
    parser.add_argument('--project_dir', type=str, help='Path to the project directory', default=None)
    parser.add_argument('--data_dir', type=str, help='Path to the data directory', default=None)
    parser.add_argument('--save_dir', type=str, help='Path to save the processed stacks', default=None)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference, e.g., "cuda:0" or "cpu"')

    args = parser.parse_args()

    if os.getenv('RUNNING_ON_SERVER') == 'true':
        project_dir = args.project_dir
        data_dir = args.data_dir
        save_dir = args.save_dir
    else:
        project_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-general\test_2_big_data_small_2_model_nameUNet3_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50"
        data_dir = r"C:\Users\rausc\Documents\EMBL\data\droso_twin-vid"
        save_dir = r"C:\Users\rausc\Documents\EMBL\data\droso_twin-vid-denoised"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    project_name = os.path.basename(project_dir)
    inference_name = os.path.basename(data_dir)
    
    results_dir = os.path.join(project_dir, 'results')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')

    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    filenames = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.tiff"))
    print("Following files will be denoised:  ", filenames)

    print(f"Using device: {device}")

    mean, std = load_normalization_params(checkpoints_dir)
    
    inf_transform = transforms.Compose([
        NormalizeInference(mean, std),
        CropToMultipleOf32Inference(),
        ToTensorInference(),
    ])

    inv_inf_transform = transforms.Compose([
        ToNumpy(),
        Denormalize(mean, std)
    ])

    inf_dataset = InferenceDatasetVideo(
        data_dir,
        transform=inf_transform
    )

    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Load hyperparameters first to get model details
    hyperparameters, epoch = load_hyperparameters(checkpoints_dir, device=device)
    model_name = hyperparameters['model_name']
    UNet_base = hyperparameters['UNet_base']

    # Dynamically get model based on model_name and UNet_base
    model = get_model(model_name, UNet_base)
    model, optimizer, epoch = load_model(checkpoints_dir, model, device=device)

    num_inf = len(inf_dataset)
    num_batch = int((num_inf / batch_size) + ((num_inf % batch_size) != 0))

    print("Starting inference")
    output_images = {}  # Dictionary to collect output images for each file

    with torch.no_grad():
        model.eval()

        for batch, data in enumerate(inf_loader):
            inputs, file_paths, slice_indices = data
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs_np = inv_inf_transform(outputs)  # Convert output tensors to numpy format for saving

            for output_img, file_path, slice_index in zip(outputs_np, file_paths, slice_indices):
                if file_path not in output_images:
                    output_images[file_path] = []
                output_images[file_path].append((slice_index, output_img))

            print('BATCH %04d/%04d' % (batch, len(inf_loader)))
    
    # Save output images
    for file_path, slices in output_images.items():
        slices.sort(key=lambda x: x[0])  # Sort by slice index
        output_stack = np.stack([img for _, img in slices], axis=0)
        filename = f'{os.path.basename(file_path)}_processed.TIFF'
        tifffile.imwrite(os.path.join(save_dir, filename), output_stack)

    print("TIFF stacks created successfully.")

if __name__ == '__main__':
    main()
