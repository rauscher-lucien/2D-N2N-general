import os
import sys
import argparse
import logging

# Add the parent directory to the system path to allow importing modules from there
sys.path.append(os.path.join(".."))

class StreamToLogger(object):
    """
    A fake file-like stream object that redirects standard output (stdout) and 
    standard error (stderr) to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        # Redirect each line of output to the logger at the specified log level
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass  # No need to handle flushing for logger redirection


# Set up logging to a file (logging.log) and configure logging options
logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append to the log file, don't overwrite
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format: timestamp, log level, message
                    level=logging.INFO,  # Log level: INFO and higher (ERROR, CRITICAL)
                    datefmt='%Y-%m-%d %H:%M:%S')  # Date format

# Also log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console output
logging.getLogger('').addHandler(console_handler)  # Add the console handler to the root logger

# Redirect standard output (stdout) and error (stderr) to the logger
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

# Import necessary modules from utils.py and train.py
from utils import *
from train import *

def main():
    # Check if the script is running on the server by checking an environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':
        # When running on the server, we expect command-line arguments to control the script's behavior
        parser = argparse.ArgumentParser(description='Process data directory.')

        # Define expected arguments for training configuration
        parser.add_argument('--train_data_dir', type=str, help='Path to the training data directory')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--train_continue', type=str, default='off', choices=['on', 'off'],
                            help='Flag to continue training: "on" (continue) or "off" (start fresh)')
        parser.add_argument('--disp_freq', type=int, default=10, help='How often to display training progress (in epochs)')
        parser.add_argument('--model_name', type=str, default='UNet3', help='Model architecture name (default: UNet3)')
        parser.add_argument('--unet_base', type=int, default=32, help='Base number of filters for the UNet model')
        parser.add_argument('--num_epoch', type=int, default=1000, help='Total number of training epochs')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer')
        parser.add_argument('--patience', type=int, default=10, help='Number of epochs with no improvement to wait before stopping early (early stopping)')

        # Parse the command-line arguments
        args = parser.parse_args()

        # Assign argument values to variables
        train_data_dir = args.train_data_dir
        project_name = args.project_name
        train_continue = args.train_continue
        disp_freq = args.disp_freq
        model_name = args.model_name
        unet_base = args.unet_base
        num_epoch = args.num_epoch
        batch_size = args.batch_size
        lr = args.lr
        patience = args.patience

        # Define the directory to store the project results and checkpoints
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'final_projects', '2D-N2N-general')
        
        # Log the training configuration for the user
        print(f"Using train data directory: {train_data_dir}")
        print(f"Train continue: {train_continue}")
        print(f"Display frequency: {disp_freq}")
        print(f"Model name: {model_name}")
        print(f"UNet base: {unet_base}")
        print(f"Number of epochs: {num_epoch}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Patience: {patience}")

    else:
        # Default settings for local testing (when running the script locally)
        train_data_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\data\big_data_small-only_mouse"
        project_dir = r"C:\Users\rausc\Documents\EMBL\final_projects\2D-N2N-general"
        project_name = 'test_x'
        train_continue = 'off'
        disp_freq = 1
        model_name = 'UNet3'
        unet_base = 64
        num_epoch = 1000
        batch_size = 8
        lr = 1e-5
        patience = 10

    # Prepare a dictionary with all the training parameters and hyperparameters
    data_dict = {
        'train_data_dir': train_data_dir,
        'project_dir': project_dir,
        'project_name': project_name,
        'disp_freq': disp_freq,
        'train_continue': train_continue,
        'hyperparameters': {
            'model_name': model_name,
            'UNet_base': unet_base,
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience
        }
    }

    # Initialize a Trainer object with the provided configuration
    trainer = Trainer(data_dict)

    # Start the training process
    trainer.train()

if __name__ == '__main__':
    # Run the main function if this script is executed directly
    main()




