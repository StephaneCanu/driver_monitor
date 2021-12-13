import torch
import torch.nn as nn


class Poseidon(nn.Module):

    def __init__(self):
        super(Poseidon, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.act1 = nn.Tanh()
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1)
        self.act2 = nn.Tanh()
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1)
        self.act3 = nn.Tanh()
        self.mp3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.act4 = nn.Tanh()
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1)
        self.act5 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=120, out_features=128)
        self.f_ac1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=128, out_features=84)
        self.f_ac2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=3)
        self.f_ac3 = nn.Tanh()

    def forward(self, x):
        x = self.mp1(self.act1(self.conv1(x)))
        x = self.mp2(self.act2(self.conv2(x)))
        x = self.mp3(self.act3(self.conv3(x)))
        x = self.act5(self.conv5(self.act4(self.conv4(x))))
        x = self.flatten(x)
        output = self.f_ac3(self.fc3(self.f_ac2(self.fc2(self.f_ac1(self.fc1(x))))))

        return output


class HeadLocModel(nn.Module):

    def __init__(self):
        super(HeadLocModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5)
        self.ac1 = nn.Tanh()
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5)
        self.ac2 = nn.Tanh()
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=4)
        self.ac3 = nn.Tanh()
        self.mp3 = nn.MaxPool2d()

        self.conv4 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3)
        self.ac4 = nn.Tanh()
        self.mp4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(in_channels=30, out_channels=120, kernel_size=3)
        self.ac5 = nn.Tanh()

        self.conv6 = nn.Conv2d(in_channels=120, out_channels=256, kernel_size=3)
        self.ac6 = nn.Tanh()

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.ac7 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc_drop1 = nn.Dropout2d(p=0.5)
        self.fc_ac1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=128, out_features=84)
        self.fc_drop2 = nn.Dropout2d(p=0.5)
        self.fc_ac2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=2)
        self.fc_drop3 = nn.Dropout2d(p=0.5)
        self.fc_ac3 = nn.Tanh()

    def forward(self, x):
        x = self.mp1(self.ac1(self.conv1(x)))
        x = self.mp2(self.ac2(self.conv2(x)))
        x = self.mp3(self.ac3(self.conv3(x)))
        x = self.mp4(self.ac4(self.conv4(x)))

        x = self.ac5(self.conv5(x))
        x = self.ac6(self.conv6(x))
        x = self.ac7(self.conv7(x))

        x = self.flatten(x)
        x = self.fc_ac1(self.fc_drop1(self.fc1(x)))
        output = self.fc_ac3(self.fc_drop3(self.fc3(self.fc_ac2(self.fc_drop2(self.ac2(x))))))
        return output





