import torch
import numpy as np


# Set the random seed manually for reproducibility.
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)





class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding= 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )
        self.conv2=torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )
        self.conv3=torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
            )
        self.conv4=torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )

        self.conv5=torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            )
        self.conv6=torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )
        self.conv7=torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            )
        self.conv8=torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
            )
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(256, 1)
            )

    # forward function
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(-1, 1 * 1 * 512)
        x = self.dense1(x)
        x = self.dense2(x)
        return x.unsqueeze(1)





