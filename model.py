import torch
import torch.nn as nn

from utils import *

class ConvBlock(nn.Module):
    ''' 
    A block that performs two sequential operations: 
    Conv2D -> BatchNorm2D -> ReLU activation.
    This block is used multiple times in the U-Net architecture.
    '''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # 3x3 convolution
            nn.BatchNorm2d(out_ch),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),  # Another 3x3 convolution
            nn.BatchNorm2d(out_ch),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )

    def forward(self, x):
        # Forward pass through the block
        return self.convblock(x)


class UNet5(nn.Module):
    """
    U-Net model with 5 levels of depth.
    It performs downsampling using max pooling and upsampling using transposed convolutions.
    """
    def __init__(self, base=32):
        super(UNet5, self).__init__()

        # Base number of filters for the first level
        self.base = base

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoding path (downsampling)
        self.enc_conv1 = ConvBlock(in_ch=1, out_ch=self.base)
        self.enc_conv2 = ConvBlock(in_ch=self.base, out_ch=2*self.base)
        self.enc_conv3 = ConvBlock(in_ch=2*self.base, out_ch=4*self.base)     
        self.enc_conv4 = ConvBlock(in_ch=4*self.base, out_ch=8*self.base)
        self.enc_conv5 = ConvBlock(in_ch=8*self.base, out_ch=16*self.base)

        # Bottleneck layer
        self.conv_b = ConvBlock(in_ch=16*self.base, out_ch=32*self.base)

        # Decoding path (upsampling)
        self.tconv5 = nn.ConvTranspose2d(in_channels=32*self.base, out_channels=16*self.base, kernel_size=2, stride=2)
        self.dec_conv5 = ConvBlock(in_ch=32*self.base, out_ch=16*self.base)
        self.tconv4 = nn.ConvTranspose2d(in_channels=16*self.base, out_channels=8*self.base, kernel_size=2, stride=2)
        self.dec_conv4 = ConvBlock(in_ch=16*self.base, out_ch=8*self.base)
        self.tconv3 = nn.ConvTranspose2d(in_channels=8*self.base, out_channels=4*self.base, kernel_size=2, stride=2)
        self.dec_conv3 = ConvBlock(in_ch=8*self.base, out_ch=4*self.base)
        self.tconv2 = nn.ConvTranspose2d(in_channels=4*self.base, out_channels=2*self.base, kernel_size=2, stride=2)
        self.dec_conv2 = ConvBlock(in_ch=4*self.base, out_ch=2*self.base)
        self.tconv1 = nn.ConvTranspose2d(in_channels=2*self.base, out_channels=self.base, kernel_size=2, stride=2)
        self.dec_conv1 = ConvBlock(in_ch=2*self.base, out_ch=self.base)

        # Final 1x1 convolution to produce the output
        self.outconv = nn.Conv2d(in_channels=self.base, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Encoding path (downsampling)
        enc_x1 = self.enc_conv1(x)
        x_p = self.pool(enc_x1)
        enc_x2 = self.enc_conv2(x_p)
        x_p = self.pool(enc_x2)
        enc_x3 = self.enc_conv3(x_p)
        x_p = self.pool(enc_x3)
        enc_x4 = self.enc_conv4(x_p)
        x_p = self.pool(enc_x4)
        enc_x5 = self.enc_conv5(x_p)
        x_p = self.pool(enc_x5)

        # Bottleneck
        x_c = self.conv_b(x_p)

        # Decoding path (upsampling)
        x5_t = self.tconv5(x_c)
        x_c = torch.cat([enc_x5, x5_t], dim=1)  # Skip connection
        dec_x5 = self.dec_conv5(x_c)
        x4_t = self.tconv4(dec_x5)
        x_c = torch.cat([enc_x4, x4_t], dim=1)  # Skip connection
        dec_x4 = self.dec_conv4(x_c)
        x3_t = self.tconv3(dec_x4)
        x_c = torch.cat([enc_x3, x3_t], dim=1)  # Skip connection
        dec_x3 = self.dec_conv3(x_c)
        x2_t = self.tconv2(dec_x3)
        x_c = torch.cat([enc_x2, x2_t], dim=1)  # Skip connection
        dec_x2 = self.dec_conv2(x_c)
        x1_t = self.tconv1(dec_x2)
        x_c = torch.cat([enc_x1, x1_t], dim=1)  # Skip connection
        dec_x1 = self.dec_conv1(x_c)

        # Final output layer
        x_final = self.outconv(dec_x1)

        return x_final


class UNet4(nn.Module):
    """
    U-Net model with 4 levels of depth.
    """
    def __init__(self, base=32):
        super(UNet4, self).__init__()

        self.base = base
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoding path (downsampling)
        self.enc_conv1 = ConvBlock(in_ch=1, out_ch=self.base)
        self.enc_conv2 = ConvBlock(in_ch=self.base, out_ch=2*self.base)
        self.enc_conv3 = ConvBlock(in_ch=2*self.base, out_ch=4*self.base)
        self.enc_conv4 = ConvBlock(in_ch=4*self.base, out_ch=8*self.base)

        # Bottleneck
        self.conv_b = ConvBlock(in_ch=8*self.base, out_ch=16*self.base)

        # Decoding path (upsampling)
        self.tconv4 = nn.ConvTranspose2d(in_channels=16*self.base, out_channels=8*self.base, kernel_size=2, stride=2)
        self.dec_conv4 = ConvBlock(in_ch=16*self.base, out_ch=8*self.base)
        self.tconv3 = nn.ConvTranspose2d(in_channels=8*self.base, out_channels=4*self.base, kernel_size=2, stride=2)
        self.dec_conv3 = ConvBlock(in_ch=8*self.base, out_ch=4*self.base)
        self.tconv2 = nn.ConvTranspose2d(in_channels=4*self.base, out_channels=2*self.base, kernel_size=2, stride=2)
        self.dec_conv2 = ConvBlock(in_ch=4*self.base, out_ch=2*self.base)
        self.tconv1 = nn.ConvTranspose2d(in_channels=2*self.base, out_channels=self.base, kernel_size=2, stride=2)
        self.dec_conv1 = ConvBlock(in_ch=2*self.base, out_ch=self.base)

        # Final 1x1 convolution to produce the output
        self.outconv = nn.Conv2d(in_channels=self.base, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Encoding path (downsampling)
        enc_x1 = self.enc_conv1(x)
        x_p = self.pool(enc_x1)
        enc_x2 = self.enc_conv2(x_p)
        x_p = self.pool(enc_x2)
        enc_x3 = self.enc_conv3(x_p)
        x_p = self.pool(enc_x3)
        enc_x4 = self.enc_conv4(x_p)
        x_p = self.pool(enc_x4)

        # Bottleneck
        x_b = self.conv_b(x_p)

        # Decoding path (upsampling)
        x4_t = self.tconv4(x_b)
        x_c = torch.cat([enc_x4, x4_t], dim=1)  # Skip connection
        dec_x4 = self.dec_conv4(x_c)
        x3_t = self.tconv3(dec_x4)
        x_c = torch.cat([enc_x3, x3_t], dim=1)  # Skip connection
        dec_x3 = self.dec_conv3(x_c)
        x2_t = self.tconv2(dec_x3)
        x_c = torch.cat([enc_x2, x2_t], dim=1)  # Skip connection
        dec_x2 = self.dec_conv2(x_c)
        x1_t = self.tconv1(dec_x2)
        x_c = torch.cat([enc_x1, x1_t], dim=1)  # Skip connection
        dec_x1 = self.dec_conv1(x_c)

        # Final output layer
        x_final = self.outconv(dec_x1)

        return x_final


class UNet3(nn.Module):
    """
    U-Net model with 3 levels of depth.
    """
    def __init__(self, base=32):
        super(UNet3, self).__init__()

        self.base = base
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoding path (downsampling)
        self.enc_conv1 = ConvBlock(in_ch=1, out_ch=self.base)
        self.enc_conv2 = ConvBlock(in_ch=self.base, out_ch=2*self.base)
        self.enc_conv3 = ConvBlock(in_ch=2*self.base, out_ch=4*self.base)

        # Bottleneck
        self.conv_b = ConvBlock(in_ch=4*self.base, out_ch=8*self.base)

        # Decoding path (upsampling)
        self.tconv3 = nn.ConvTranspose2d(in_channels=8*self.base, out_channels=4*self.base, kernel_size=2, stride=2)
        self.dec_conv3 = ConvBlock(in_ch=8*self.base, out_ch=4*self.base)
        self.tconv2 = nn.ConvTranspose2d(in_channels=4*self.base, out_channels=2*self.base, kernel_size=2, stride=2)
        self.dec_conv2 = ConvBlock(in_ch=4*self.base, out_ch=2*self.base)
        self.tconv1 = nn.ConvTranspose2d(in_channels=2*self.base, out_channels=self.base, kernel_size=2, stride=2)
        self.dec_conv1 = ConvBlock(in_ch=2*self.base, out_ch=self.base)

        # Final 1x1 convolution to produce the output
        self.outconv = nn.Conv2d(in_channels=self.base, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Encoding path (downsampling)
        enc_x1 = self.enc_conv1(x)
        x_p = self.pool(enc_x1)
        enc_x2 = self.enc_conv2(x_p)
        x_p = self.pool(enc_x2)
        enc_x3 = self.enc_conv3(x_p)
        x_p = self.pool(enc_x3)

        # Bottleneck
        x_b = self.conv_b(x_p)

        # Decoding path (upsampling)
        x3_t = self.tconv3(x_b)
        x_c = torch.cat([enc_x3, x3_t], dim=1)  # Skip connection
        dec_x3 = self.dec_conv3(x_c)
        x2_t = self.tconv2(dec_x3)
        x_c = torch.cat([enc_x2, x2_t], dim=1)  # Skip connection
        dec_x2 = self.dec_conv2(x_c)
        x1_t = self.tconv1(dec_x2)
        x_c = torch.cat([enc_x1, x1_t], dim=1)  # Skip connection
        dec_x1 = self.dec_conv1(x_c)

        # Final output layer
        x_final = self.outconv(dec_x1)

        return x_final


