# models/gan_model.py
import numpy as np
import torch
import os
from torch import nn
from torchvision import transforms
from PIL import Image
from app.config import CHECKPOINT_PATH

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=64):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, apply_batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, apply_dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder (Downsampling layers)
        self.down1 = down_block(input_nc, ngf, apply_batchnorm=False)
        self.down2 = down_block(ngf, ngf * 2)
        self.down3 = down_block(ngf * 2, ngf * 4)
        self.down4 = down_block(ngf * 4, ngf * 8)
        self.down5 = down_block(ngf * 8, ngf * 8)
        self.down6 = down_block(ngf * 8, ngf * 8)
        self.down7 = down_block(ngf * 8, ngf * 8)
        self.down8 = down_block(ngf * 8, ngf * 8, apply_batchnorm=False)

        # Decoder (Upsampling layers)
        self.up1 = up_block(ngf * 8, ngf * 8, apply_dropout=True)
        self.up2 = up_block(ngf * 8 * 2, ngf * 8, apply_dropout=True)
        self.up3 = up_block(ngf * 8 * 2, ngf * 8, apply_dropout=True)
        self.up4 = up_block(ngf * 8 * 2, ngf * 8)
        self.up5 = up_block(ngf * 8 * 2, ngf * 4)
        self.up6 = up_block(ngf * 4 * 2, ngf * 2)
        self.up7 = up_block(ngf * 2 * 2, ngf)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))


# PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64):
        super(PatchGANDiscriminator, self).__init__()

        def disc_block(in_channels, out_channels, stride=2, apply_batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(input_nc + output_nc, ndf, apply_batchnorm=False),
            disc_block(ndf, ndf * 2),
            disc_block(ndf * 2, ndf * 4),
            disc_block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, sketch):
        input = torch.cat((img, sketch), 1)  # Concatenate input and output
        return self.model(input)


# Loss Functions
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)

def load_trained_model(checkpoint_path=CHECKPOINT_PATH):
    """Loads the trained Pix2Pix generator model."""
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetGenerator(output_nc=1).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()
    return generator

def preprocess_image(image_path, device):
    """Loads and preprocesses an image for the model."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0).to(device)

def postprocess_image(tensor):
    """Converts a model output tensor (grayscale) back to a displayable image."""
    tensor = tensor.squeeze().detach().cpu().numpy()  # Fix: Only one squeeze()
    tensor = (tensor * 0.5) + 0.5  # Undo normalization
    return np.clip(tensor * 255, 0, 255).astype(np.uint8)
