import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Define a basic CNN model for regression
class CNNModel_4(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNModel_4, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 40 * 40, out_features=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class CNNModel_4_2(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNModel_4_2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 40 * 40, out_features=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
class CNNModel_1(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNModel_1, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Update the input feature size for the fully connected layer
        # (assuming input images are 640x640 and pooling reduces by a factor of 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 320 * 320, out_features=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class CNNModel_2(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNModel_2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Update the input feature size for the fully connected layer
        # (assuming input images are 640x640 and pooling reduces by a factor of 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 160 * 160, out_features=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class CNNModel_2_2(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNModel_2_2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Update the input feature size for the fully connected layer
        # (assuming input images are 640x640 and pooling reduces by a factor of 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16* 160 * 160, out_features=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
class CNNModel_3(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNModel_3, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Update the input feature size for the fully connected layer
        # (assuming input images are 640x640 and pooling reduces by a factor of 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 80 * 80, out_features=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
