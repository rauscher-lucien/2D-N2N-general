import os
import warnings
import glob
import random
import torch
import numpy as np
import tifffile
import pickle
import matplotlib.pyplot as plt

def create_result_dir(project_dir, project_name, hyperparameters, train_data_dir):
    """
    Creates directories for storing results and checkpoints based on project name, 
    training data directory, and hyperparameters. It helps to keep results organized 
    by project and training configuration.
    
    Parameters:
    - project_dir: Base directory for the project.
    - project_name: Name of the project.
    - hyperparameters: Dictionary of hyperparameters used in training.
    - train_data_dir: Directory containing the training data.
    
    Returns:
    - results_dir: Directory where training results will be saved.
    - checkpoints_dir: Directory where model checkpoints will be saved.
    """
    # Extract the base name of the training data directory
    base_name = os.path.basename(train_data_dir)
    # Create a string from the hyperparameters to use in directory names
    hyperparams_str = '_'.join([f"{key}{value}" for key, value in hyperparameters.items()])
    # Combine the project name, base name, and hyperparameters to form the result directory name
    name = f"{project_name}_{base_name}_{hyperparams_str}"

    # Create result and checkpoint directories
    results_dir = os.path.join(project_dir, name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    return results_dir, checkpoints_dir

def create_train_dir(results_dir):
    """
    Creates a directory for saving intermediate training results.
    
    Parameters:
    - results_dir: Directory where results will be saved.
    
    Returns:
    - train_dir: Directory for saving training data like logs or model outputs.
    """
    train_dir = os.path.join(results_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    return train_dir

def compute_global_mean_and_std(dataset_path, checkpoints_path):
    """
    Computes and saves the global mean and standard deviation across all TIFF stacks in a directory.
    The computed values are stored in a pickle file for later use.
    
    Parameters:
    - dataset_path: Path to the directory containing TIFF files.
    - checkpoints_path: Path where the mean and standard deviation should be saved.
    
    Returns:
    - global_mean: The mean pixel intensity across all images.
    - global_std: The standard deviation of pixel intensities across all images.
    """
    # Define the path to save the normalization parameters
    save_path = os.path.join(checkpoints_path, 'normalization_params.pkl')

    # If the normalization parameters already exist, load them
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            params = pickle.load(f)
            global_mean = params['mean']
            global_std = params['std']
        print(f"Loaded global mean and std parameters from {save_path}")
    else:
        # Compute mean and std across all TIFF files in the dataset
        all_means = []
        all_stds = []
        for subdir, _, files in os.walk(dataset_path):
            for filename in files:
                if filename.lower().endswith(('.tif', '.tiff')):
                    filepath = os.path.join(subdir, filename)
                    stack = tifffile.imread(filepath)
                    all_means.append(np.mean(stack))
                    all_stds.append(np.std(stack))
                    
        global_mean = np.mean(all_means)
        global_std = np.mean(all_stds)
        
        # Save the computed values to a pickle file for future use
        with open(save_path, 'wb') as f:
            pickle.dump({'mean': global_mean, 'std': global_std}, f)
        
        print(f"Global mean and std parameters saved to {save_path}")

    return global_mean, global_std

def load_normalization_params(data_dir):
    """
    Loads the mean and standard deviation values from a pickle file located in the specified directory.
    
    Parameters:
    - data_dir: Path to the directory containing the 'normalization_params.pkl' file.
    
    Returns:
    - mean: The mean value from the normalization parameters.
    - std: The standard deviation value from the normalization parameters.
    """
    load_path = os.path.join(data_dir, 'normalization_params.pkl')
    
    # Load the saved mean and std values
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    mean = params['mean']
    std = params['std']
    
    return mean, std

def print_tiff_filenames(root_folder_path):
    """
    Prints the filenames of all TIFF files in the specified folder and its subdirectories.
    
    Parameters:
    - root_folder_path: Path to the folder containing TIFF stack files.
    """
    for subdir, _, files in os.walk(root_folder_path):
        # Sort the files to ensure consistency in order
        sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
        for filename in sorted_files:
            print(filename)

def get_device():
    """
    Determines whether a GPU (CUDA) is available and returns the appropriate device for PyTorch.
    
    Returns:
    - device: Either 'cuda' if a GPU is available, or 'cpu' if not.
    """
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    
    return device

def crop_tiff_depth_to_divisible(path, divisor):
    """
    Crops the depth (number of slices) of each TIFF file in the specified directory to ensure that 
    it is divisible by the given divisor. This can be useful for batch processing in machine learning models.
    
    Parameters:
    - path: Directory containing the TIFF files to be processed.
    - divisor: The number that the depth should be divisible by.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, file)
                with tifffile.TiffFile(file_path) as tif:
                    images = tif.asarray()
                    depth = images.shape[0]
                    
                    # If the depth is not divisible by the divisor, crop it
                    if depth % divisor != 0:
                        new_depth = depth - (depth % divisor)
                        cropped_images = images[:new_depth]
                        
                        # Save the cropped TIFF stack
                        tifffile.imwrite(file_path, cropped_images, photometric='minisblack')
                        print(f'Cropped and saved: {file_path}')

