import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Normalize(object):
    """
    Normalize both input and target images using a specified mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean value used for normalization.
        std (float or tuple): Standard deviation used for normalization.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Apply normalization to the input and target images.
        
        Args:
            data (tuple): A tuple containing the input and target images.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data
        # Normalize the input and target images
        input_normalized = (input_img - self.mean) / self.std
        target_normalized = (target_img - self.mean) / self.std

        return input_normalized, target_normalized
    

class NormalizeInference(object):
    """
    Normalize input image for inference using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean value used for normalization.
        std (float or tuple): Standard deviation used for normalization.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Normalize the input image (used during inference).
        
        Args:
            data: The input image.
        
        Returns:
            Normalized input image.
        """
        input_img = data
        input_normalized = (input_img - self.mean) / self.std
        return input_normalized


class RandomHorizontalFlip:
    """
    Randomly flip the input and target images horizontally with a 50% chance.
    This transformation is typically used during training for data augmentation.
    """

    def __call__(self, data):
        """
        Apply horizontal flipping to both input and target images.
        
        Args:
            data (tuple): A tuple containing the input stack and target slice.
        
        Returns:
            Tuple: Input and target images, possibly flipped horizontally.
        """
        input_stack, target_slice = data

        # Flip images with a probability of 50%
        if np.random.rand() > 0.5:
            input_stack = np.flip(input_stack, axis=1)  # Flip horizontally (axis 1)
            target_slice = np.flip(target_slice, axis=1)

        return input_stack, target_slice


class RandomCrop:
    """
    Randomly crop the input stack and target slice to the specified output size.
    This transformation is used during training for data augmentation.
    
    Args:
        output_size (tuple): The desired output size (height, width).
    """

    def __init__(self, output_size=(64, 64)):
        self.output_size = output_size

    def __call__(self, data):
        """
        Crop both input and target images randomly.
        
        Args:
            data (tuple): A tuple containing the input stack and target slice.
        
        Returns:
            Tuple: Cropped input and target images.
        """
        input_stack, target_slice = data

        h, w = input_stack.shape
        new_h, new_w = self.output_size

        # Randomly choose the top-left corner of the crop
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Crop the images
        input_cropped = input_stack[top:top+new_h, left:left+new_w]
        target_cropped = target_slice[top:top+new_h, left:left+new_w]

        return input_cropped, target_cropped


class CropToMultipleOf32Inference(object):
    """
    Crop the input image to ensure its dimensions are multiples of 32.
    This is useful for neural networks that require input dimensions divisible by 32.
    """

    def __call__(self, data):
        """
        Crop the input image to dimensions divisible by 32.
        
        Args:
            data (numpy.ndarray): Input image.
        
        Returns:
            Cropped input image.
        """
        input_slice = data
        h, w = data.shape

        # Adjust height and width to be divisible by 32
        new_h = h - (h % 32)
        new_w = w - (w % 32)

        # Crop the image symmetrically
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image using the computed indices
        input_slice_cropped = input_slice[top:top+new_h, left:left+new_w]

        return input_slice_cropped


class CropToMultipleOf16Inference(object):
    """
    Crop the input image to ensure its dimensions are multiples of 16.
    This version is similar to the 32 version but for dimensions divisible by 16.
    """

    def __call__(self, data):
        """
        Crop the input image to dimensions divisible by 16.
        
        Args:
            data (numpy.ndarray): Input image.
        
        Returns:
            Cropped input image.
        """
        input_slice = data
        h, w = data.shape

        # Adjust height and width to be divisible by 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Crop the image symmetrically
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image using the computed indices
        input_slice_cropped = input_slice[top:top+new_h, left:left+new_w]

        return input_slice_cropped


class ToTensor(object):
    """
    Convert a NumPy array to a PyTorch tensor.
    This transformation is typically used at the end of a pipeline to prepare the data for the model.
    """

    def __call__(self, data):
        """
        Convert the input and target images to PyTorch tensors.
        
        Args:
            data (tuple): A tuple containing the input and target images.
        
        Returns:
            Tuple: Input and target images as tensors.
        """
        def convert_image(img):
            return torch.from_numpy(img.astype(np.float32))  # Convert NumPy array to PyTorch tensor
        return tuple(convert_image(img) for img in data)


class ToTensorInference(object):
    """
    Convert a single NumPy image to a PyTorch tensor for inference.
    """

    def __call__(self, img):
        # Convert the input image to a PyTorch tensor
        return torch.from_numpy(img.astype(np.float32))


class ToNumpy(object):
    """
    Convert a PyTorch tensor back to a NumPy array.
    Typically used for post-processing model outputs.
    """

    def __call__(self, data):
        # Convert tensor to NumPy array and transpose to [H, W, C] format
        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)


class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its minimum and maximum values.
    Useful when the input has an arbitrary range.
    """

    def __call__(self, tensor):
        """
        Normalize the tensor to [0, 1] range based on its dynamic range.
        
        Args:
            tensor: A tensor with arbitrary range.
        
        Returns:
            Tensor: Normalized tensor in the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()

        # Avoid division by zero
        if (max_val - min_val).item() > 0:
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, return a tensor filled with zeros
            normalized_tensor = tensor.clone().fill_(0)

        return normalized_tensor


class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation and convert it to 16-bit format.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation values.
        
        Args:
            mean (float or tuple): Mean for normalization.
            std (float or tuple): Standard deviation for normalization.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            Denormalized 16-bit image.
        """
        # Reverse the normalization
        img_denormalized = (img * self.std) + self.mean

        # Convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)

        return img_16bit

