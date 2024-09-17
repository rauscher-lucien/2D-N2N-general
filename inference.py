import os
import sys
sys.path.append(os.path.join(".."))  # Add the parent directory to the system path for importing modules

import torch
import numpy as np
import tifffile
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *  # Importing model architectures (e.g., UNet3, UNet4, UNet5)
from transforms import *  # Importing data transformation utilities
from utils import *  # Importing utility functions
from dataset import *  # Importing dataset class

def load_hyperparameters(checkpoints_dir, device='cpu'):
    """
    Loads the hyperparameters and epoch number from the checkpoint file.
    
    Parameters:
    - checkpoints_dir: Directory where the model checkpoint is stored.
    - device: Device to map the loaded checkpoint to (CPU or GPU).
    
    Returns:
    - hyperparameters: Dictionary containing the model's hyperparameters.
    - epoch: The epoch number at which the model was saved.
    """
    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    dict_net = torch.load(checkpoint_path, map_location=device)
    hyperparameters = dict_net['hyperparameters']  # Load hyperparameters
    epoch = dict_net['epoch']  # Load epoch number

    return hyperparameters, epoch

def load_model(checkpoints_dir, model, optimizer=None, device='cpu'):
    """
    Loads a saved model and its optimizer state from the checkpoint.
    
    Parameters:
    - checkpoints_dir: Directory where the model checkpoint is stored.
    - model: The model architecture to load the state into.
    - optimizer: The optimizer to load its state into (if any).
    - device: Device to map the loaded model to (CPU or GPU).
    
    Returns:
    - model: The model with loaded weights.
    - optimizer: The optimizer with loaded state.
    - epoch: The epoch number at which the model was saved.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  # Default optimizer if not provided

    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
    dict_net = torch.load(checkpoint_path, map_location=device)

    # Load model weights and optimizer state
    model.load_state_dict(dict_net['model'])
    optimizer.load_state_dict(dict_net['optimizer'])
    epoch = dict_net['epoch']  # Load the saved epoch number

    model.to(device)  # Move model to the appropriate device (CPU/GPU)

    print(f'Loaded {epoch}th network with hyperparameters: {dict_net["hyperparameters"]}')

    return model, optimizer, epoch

def get_model(model_name, UNet_base):
    """
    Returns the model architecture based on the provided model name and base filter size.
    
    Parameters:
    - model_name: Name of the model architecture (e.g., 'UNet3', 'UNet4', 'UNet5').
    - UNet_base: Base filter size for the U-Net model.
    
    Returns:
    - The initialized model instance.
    """
    if model_name == 'UNet3':
        return UNet3(base=UNet_base)
    elif model_name == 'UNet4':
        return UNet4(base=UNet_base)
    elif model_name == 'UNet5':
        return UNet5(base=UNet_base)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():

    # Paths for the project directory and data for inference
    project_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-general\test_2_big_data_small_2_model_nameUNet4_UNet_base32_num_epoch1000_batch_size8_lr1e-05_patience50"
    data_dir = r"\\tier2.embl.de\prevedel\members\Wang\Data\Mouse\Embryo\20230615\LogScale\Mouse_Embryo_10h"
    
    # Extract the inference name and project name
    inference_name = os.path.basename(data_dir)
    project_name = os.path.basename(project_dir)
    method_name = os.path.basename(os.path.dirname(project_dir))

    # Paths for saving results and loading checkpoints
    results_dir = os.path.join(project_dir, 'results')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')

    # Make a folder to store inference results
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)
    
    ## Load image stack for inference
    filenames = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.tiff"))
    print("Following files will be denoised:  ", filenames)

    # Check if GPU is accessible
    if torch.cuda.is_available():
        print("\nGPU will be used.")
        device = torch.device("cuda:0")
    else:
        print("\nCPU will be used.")
        device = torch.device("cpu")

    # Load normalization parameters (mean and std) from saved checkpoint
    mean, std = load_normalization_params(checkpoints_dir)
    
    # Define transformations for inference
    inf_transform = transforms.Compose([
        NormalizeInference(mean, std),  # Normalize based on mean and std
        CropToMultipleOf32Inference(),  # Ensure dimensions are divisible by 32
        ToTensorInference(),  # Convert images to PyTorch tensors
    ])

    # Define inverse transformations for post-processing
    inv_inf_transform = transforms.Compose([
        ToNumpy(),  # Convert tensors back to NumPy arrays
        Denormalize(mean, std)  # Denormalize using saved mean and std
    ])

    # Create inference dataset
    inf_dataset = InferenceDataset(
        data_dir,
        transform=inf_transform
    )

    # Create a DataLoader for batching the inference data
    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2  # Number of subprocesses for data loading
    )

    # Load hyperparameters to get model details
    hyperparameters, epoch = load_hyperparameters(checkpoints_dir, device=device)
    model_name = hyperparameters['model_name']
    UNet_base = hyperparameters['UNet_base']

    # Dynamically get model based on model_name and UNet_base
    model = get_model(model_name, UNet_base)
    
    # Load model and optimizer from checkpoint
    model, optimizer, epoch = load_model(checkpoints_dir, model, device=device)

    num_inf = len(inf_dataset)  # Total number of images for inference
    num_batch = int((num_inf / batch_size) + ((num_inf % batch_size) != 0))  # Number of batches

    print("Starting inference")
    output_images = []  # List to store the output images

    # Run inference with the model
    with torch.no_grad():  # Disable gradient calculation (inference only)
        model.eval()  # Set model to evaluation mode

        for batch, data in enumerate(inf_loader):
            input_img = data.to(device)  # Move input data to the correct device (CPU or GPU)

            # Run model inference
            output_img = model(input_img)
            output_img_np = inv_inf_transform(output_img)  # Convert output tensors to NumPy arrays for saving

            # Append each processed image to the output list
            for img in output_img_np:
                output_images.append(img)

            print('BATCH %04d/%04d' % (batch, len(inf_loader)))  # Display progress
    
    # Stack and save the output images into a TIFF file
    output_stack = np.stack(output_images, axis=0)
    filename = f'{method_name}_output_stack-{inference_name}-project-{project_name}-epoch{epoch}.TIFF'
    tifffile.imwrite(os.path.join(inference_folder, filename), output_stack)

    print("TIFF stacks created successfully.")

if __name__ == '__main__':
    main()


