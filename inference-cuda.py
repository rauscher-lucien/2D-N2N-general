import os
import sys
import argparse
import logging
import glob
import torch
import numpy as np
import time
from torchvision import transforms

# Add the parent directory to the system path for importing modules from there
sys.path.append(os.path.join(".."))

# Import necessary modules for model, transformations, utilities, and dataset
from model import *
from transforms import *
from utils import *
from dataset import *

class StreamToLogger(object):
    """
    Redirects stdout and stderr to a logging instance instead of printing to the console.
    This class captures standard output and error messages and logs them using Python's logging module.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        # Log each line in the buffer
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass  # No need to flush when redirecting to a logger

def setup_logging(log_file='logging.log'):
    """
    Set up logging to a file and console output. It redirects stdout and stderr to the logger.
    
    Args:
        log_file: The file to which the logs will be written.
    """
    logging.basicConfig(
        filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()  # Console handler for logging
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def load_model(checkpoints_dir, model, optimizer=None, device='cpu'):
    """
    Load the model and optimizer state from a checkpoint.
    
    Args:
        checkpoints_dir: Directory where the model checkpoint is stored.
        model: The model architecture to load the state into.
        optimizer: The optimizer to load its state into (default is Adam optimizer).
        device: The device (CPU or GPU) to load the model onto.

    Returns:
        model: The model with loaded weights.
        optimizer: The optimizer with loaded state.
        epoch: The epoch number at which the model was saved.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  # Use Adam optimizer if not provided

    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')  # Path to the model checkpoint
    dict_net = torch.load(checkpoint_path, map_location=device)

    # Load the saved model and optimizer state
    model.load_state_dict(dict_net['model'])
    optimizer.load_state_dict(dict_net['optimizer'])
    epoch = dict_net['epoch']

    model.to(device)  # Move model to the appropriate device

    print(f'Loaded {epoch}th network with hyperparameters: {dict_net["hyperparameters"]}')
    return model, optimizer, epoch

def load_hyperparameters(checkpoints_dir, device='cpu'):
    """
    Load hyperparameters and epoch number from a model checkpoint.
    
    Args:
        checkpoints_dir: Directory where the model checkpoint is stored.
        device: The device (CPU or GPU) to map the checkpoint to.

    Returns:
        hyperparameters: Dictionary of hyperparameters used for the model.
        epoch: The epoch number at which the model was saved.
    """
    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    dict_net = torch.load(checkpoint_path, map_location=device)
    hyperparameters = dict_net['hyperparameters']  # Load hyperparameters
    epoch = dict_net['epoch']  # Load epoch number

    return hyperparameters, epoch

def get_model(model_name, UNet_base):
    """
    Return the correct model architecture based on the provided model name and base filter size.
    
    Args:
        model_name: Name of the model architecture (e.g., 'UNet3', 'UNet4', 'UNet5').
        UNet_base: Base number of filters in the UNet architecture.

    Returns:
        The initialized model instance.
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
    setup_logging()  # Set up logging for the script

    # Set up argument parsing for command-line inputs
    parser = argparse.ArgumentParser(description='Process inference parameters.')
    parser.add_argument('--project_dir', type=str, help='Path to the project directory', default=None)
    parser.add_argument('--data_dir', type=str, help='Path to the data directory', default=None)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference, e.g., "cuda:0" or "cpu"')

    args = parser.parse_args()

    # If running on the server, get project and data directories from the environment variables
    if os.getenv('RUNNING_ON_SERVER') == 'true':
        project_dir = args.project_dir
        data_dir = args.data_dir
    else:
        # Use default directories if running locally
        project_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\final_projects\2D-N2N-general\test_3_big_data_small_2_model_nameUNet5_UNet_base16_num_epoch1000_batch_size8_lr1e-05_patience50"
        data_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\data\big_data_small-test\mouse"

    # Set up the device for inference (GPU if available, otherwise CPU)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Extract names for organizing results
    project_name = os.path.basename(project_dir)
    method_name = os.path.basename(os.path.dirname(project_dir))
    inference_name = os.path.basename(data_dir)
    
    # Set up directories for results and checkpoints
    results_dir = os.path.join(project_dir, 'results')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')

    # Find all TIFF files in the data directory for inference
    filenames = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.tiff"))
    print("Following files will be denoised:  ", filenames)

    print(f"Using device: {device}")

    # Load normalization parameters (mean and standard deviation) from the checkpoint
    mean, std = load_normalization_params(checkpoints_dir)
    
    # Set up transformations for inference (normalization, cropping, conversion to tensor)
    inf_transform = transforms.Compose([
        NormalizeInference(mean, std),
        CropToMultipleOf32Inference(),
        ToTensorInference(),
    ])

    # Set up inverse transformations to convert tensors back to images for saving
    inv_inf_transform = transforms.Compose([
        ToNumpy(),
        Denormalize(mean, std)
    ])

    # Create the dataset for inference
    inf_dataset = InferenceDataset(
        data_dir,
        transform=inf_transform
    )

    # Set up the data loader for batching the dataset during inference
    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2  # Number of subprocesses for data loading
    )

    # Load hyperparameters and model from checkpoint
    hyperparameters, epoch = load_hyperparameters(checkpoints_dir, device=device)
    model_name = hyperparameters['model_name']
    UNet_base = hyperparameters['UNet_base']

    # Initialize the model
    model = get_model(model_name, UNet_base)
    model, optimizer, epoch = load_model(checkpoints_dir, model, device=device)

    # Inference loop, including timing
    num_inf = len(inf_dataset)
    num_batch = int((num_inf / batch_size) + ((num_inf % batch_size) != 0))

    inference_times = []  # To store inference times

    print("Starting inference")

    # Perform inference 10 times for timing
    for i in range(10):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # Start timing

            with torch.no_grad():  # Disable gradient calculation during inference
                model.eval()  # Set model to evaluation mode
                for batch, data in enumerate(inf_loader):
                    input_img = data.to(device)  # Move input to the correct device
                    output_img = model(input_img)  # Run the model
                    output_img_np = inv_inf_transform(output_img)  # Convert output to NumPy for saving

            end_event.record()  # End timing
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        else:
            # CPU timing if GPU is not available
            start_time = time.time()

            with torch.no_grad():
                model.eval()
                for batch, data in enumerate(inf_loader):
                    input_img = data.to(device)
                    output_img = model(input_img)
                    output_img_np = inv_inf_transform(output_img)

            inference_time = time.time() - start_time  # Calculate elapsed time

        inference_times.append(inference_time)
        print(f"Inference {i+1} Time: {inference_time} seconds")
        logging.info(f"Inference {i+1} Time: {inference_time} seconds")
    
    # Calculate and log the average inference time
    avg_inference_time = np.mean(inference_times)
    print(f"Average Inference Time: {avg_inference_time} seconds")
    logging.info(f"Average Inference Time: {avg_inference_time} seconds")

if __name__ == '__main__':
    main()


