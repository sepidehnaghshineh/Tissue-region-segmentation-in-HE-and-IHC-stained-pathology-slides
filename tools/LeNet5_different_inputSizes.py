import torch
import torch.nn as nn

###############################################################################################################
class LeNet5_128(nn.Module):
    def __init__(self):
        super(LeNet5_128, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Adjust padding for 128x128 input
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # Adjust padding for 128x128 input
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output after convolution and pooling
        self.fc1_input_size = 16 * 32 * 32  # Adjust based on the new output size
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Adjust based on the new output size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


###############################################################################################################
class LeNet5_64(nn.Module):
    def __init__(self):
        super(LeNet5_64, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Adjust padding for 64x64 input
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # Adjust padding for 64x64 input
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output after convolution and pooling
        self.fc1_input_size = 16 * 16 * 16  # Adjust based on the new output size
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Adjust based on the new output size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
###############################################################################################################
class LeNet5_32(nn.Module):
    def __init__(self):
        super(LeNet5_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Adjust padding for 32x32 input
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # Adjust padding for 32x32 input
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output after convolution and pooling
        self.fc1_input_size = 16 * 8 * 8  # Adjust based on the new output size
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Adjust based on the new output size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
###############################################################################################################
class LeNet5_16(nn.Module):
    def __init__(self):
        super(LeNet5_16, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Adjust padding for 16x16 input
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # Adjust padding for 16x16 input
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output after convolution and pooling
        self.fc1_input_size = 16 * 4 * 4  # Adjust based on the new output size
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Adjust based on the new output size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################################################################
###############################################################################################################

# Function to select the appropriate model based on input size
def select_model(input_size):
    if input_size == 128:
        return LeNet5_128()
    elif input_size == 64:
        return LeNet5_64()
    elif input_size == 32:
        return LeNet5_32()
    elif input_size == 16:
        return LeNet5_16()
    else:
        raise ValueError("Unsupported input size")



class LeNet5_128_with_hue(nn.Module):
    def __init__(self):
        super(LeNet5_128, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Adjust padding for 128x128 input
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # Adjust padding for 128x128 input
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output after convolution and pooling
        self.fc1_input_size = 16 * 32 * 32  # Adjust based on the new output size
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Adjust based on the new output size
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x