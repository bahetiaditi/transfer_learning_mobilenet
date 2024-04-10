# models/segnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up1 = self.upsample_block(1280, 512)
        self.refine1 = self.refinement_block(512, 512)

        self.up2 = self.upsample_block(512, 256)
        self.refine2 = self.refinement_block(256, 256)

        self.up3 = self.upsample_block(256, 128)
        self.refine3 = self.refinement_block(128, 128)

        self.up4 = self.upsample_block(128, 64)
        self.refine4 = self.refinement_block(64, 64)


        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def refinement_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.refine1(x)
        x = self.up2(x)
        x = self.refine2(x)
        x = self.up3(x)
        x = self.refine3(x)
        x = self.up4(x)
        x = self.refine4(x)
        x = self.up5(x)
        x = self.final_conv(x)
        return x

class SegNet(nn.Module):
    def __init__(self, pretrained=True, freeze_encoder=True):
        super(SegNet, self).__init__()
        self.encoder = models.mobilenet_v2(pretrained=pretrained).features
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 
