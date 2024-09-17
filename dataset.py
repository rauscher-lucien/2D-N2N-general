import os
import numpy as np
import torch
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class TwoSliceDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading pairs of consecutive slices from 3D TIFF volumes.
    This is used for training where we treat one slice as the input and the next as the target.
    """
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # Dictionary to store the loaded 3D volumes in memory
        self.pairs = self.preload_and_make_pairs(root_folder_path)  # Preload volumes and create slice pairs

    def preload_and_make_pairs(self, root_folder_path):
        """
        This function loads all TIFF files in the specified folder and its subdirectories.
        It creates a list of tuples where each tuple contains the path to the TIFF file and 
        the indices of consecutive slices that will be used as input-target pairs.
        """
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])  # Sort files to ensure proper order
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)  # Load the 3D volume (TIFF stack)
                self.preloaded_data[full_path] = volume  # Store the volume in the preloaded_data dictionary
                num_slices = volume.shape[0]  # Number of slices in the volume
                for i in range(num_slices - 1):  # Create pairs from consecutive slices
                    input_slice_index = i
                    target_slice_index = i + 1
                    pairs.append((full_path, input_slice_index, target_slice_index))  # Add pair of indices
        return pairs

    def __len__(self):
        # Return the total number of slice pairs in the dataset
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Retrieve a specific input-target pair by index. The data is loaded from memory (preloaded_data),
        and the necessary transformations are applied if specified.
        """
        file_path, input_slice_index, target_slice_index = self.pairs[index]
        
        # Get the input and target slices from the preloaded data
        input_slice = self.preloaded_data[file_path][input_slice_index]
        target_slice = self.preloaded_data[file_path][target_slice_index]

        # Apply transformations if any are provided
        if self.transform:
            input_slice, target_slice = self.transform((input_slice, target_slice))

        # Add an extra dimension to represent the channel (required by the model)
        input_slice = input_slice[np.newaxis, ...]
        target_slice = target_slice[np.newaxis, ...]

        return input_slice, target_slice


class InferenceDataset(torch.utils.data.Dataset):
    """
    Dataset class for inference, where we only need the individual slices from the first loaded volume.
    The difference from TwoSliceDataset is that we don't pair consecutive slices for training.
    """
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # Dictionary to store the loaded 3D volume in memory
        self.slices = self.preload_first_volume(root_folder_path)  # Preload the first volume and create a list of slices

    def preload_first_volume(self, root_folder_path):
        """
        This function loads only the first 3D volume (TIFF stack) found in the folder.
        It creates a list of tuples where each tuple contains the path to the file and 
        the index of a single slice. This is useful for inference tasks.
        """
        slices = []
        volume_loaded = False  # Flag to stop after loading the first volume

        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])  # Sort files to ensure proper order
            for f in sorted_files:
                if volume_loaded:
                    break  # Stop once the first volume is loaded
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)  # Load the 3D volume (TIFF stack)
                self.preloaded_data[full_path] = volume  # Store the volume in the preloaded_data dictionary
                num_slices = volume.shape[0]  # Number of slices in the volume
                for i in range(num_slices):  # Add all slices from the first volume to the list
                    slices.append((full_path, i))
                volume_loaded = True  # Set the flag to indicate that the first volume is loaded
            if volume_loaded:
                break  # Exit the outer loop once the first volume is loaded

        return slices

    def __len__(self):
        # Return the total number of slices in the first loaded volume
        return len(self.slices)

    def __getitem__(self, index):
        """
        Retrieve a specific slice by index. The data is loaded from memory (preloaded_data),
        and the necessary transformations are applied if specified.
        """
        file_path, slice_index = self.slices[index]
        
        # Get the slice from the preloaded data
        input_slice = self.preloaded_data[file_path][slice_index]

        # Apply transformation if provided
        if self.transform:
            input_slice = self.transform(input_slice)
        
        # Add an extra dimension to represent the channel (required by the model)
        input_slice = input_slice[np.newaxis, ...]

        return input_slice



