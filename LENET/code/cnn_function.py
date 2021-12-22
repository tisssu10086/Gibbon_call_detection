import torch
import numpy as np


# Set the random seed manually for reproducibility.
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




class LENET(torch.nn.Module):
    def __init__(self):
        super(LENET, self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding= 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )
        self.conv2=torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 64, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
            )
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
            )
        self.dense3 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1)
            )

    # forward function
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 8 * 8 * 64)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x.unsqueeze(1)







