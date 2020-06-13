from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features))

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        self.input_conv = nn.Sequential(
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(3, 64, 7),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))

        # Downsampling
        self.downsampling = nn.Sequential(
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(inplace=True))

        # Residual blocks
        self.res = []
        for _ in range(n_residual_blocks):
            self.res += [ResidualBlock(256)]
        self.residual_blocks = nn.Sequential(*self.res)

        # Upsampling
        self.upsampling = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))

        # Output layer
        self.output = nn.Sequential(
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(64, 3, 7),
                        nn.Tanh())

    def forward(self, x):
        x = self.input_conv(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        self.features_extractor = nn.Sequential(
                    nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True))

        # FCN classification layer
        self.classificator = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classificator(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)