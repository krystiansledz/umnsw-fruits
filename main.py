import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPyTorchModel(nn.Module):
    def __init__(self, input_shape, outputs_number):
        super(CustomPyTorchModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=48, kernel_size=11, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(48, 128, 5, padding='same')
        self.conv3 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv4 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv5 = nn.Conv2d(256, 128, 3, padding='same')

        # Calculate the size of the flattened features after the conv and pool layers
        with torch.no_grad():
            self._to_linear = None
            self.convs(torch.zeros(1, *input_shape))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, outputs_number)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def get_pytorch_model(input_shape, outputs_number, lr):
    model = CustomPyTorchModel(input_shape, outputs_number)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    return model, optimizer, loss_function


def train_model(model, train_loader, val_loader, optimizer, loss_function, epochs=10):
    for epoch in range(epochs):
        print("Start of epoch {}".format(epoch))
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total}%")

# Training and validation loop
transform = transforms.Compose([
    transforms.Resize((52, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(root='52x64', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, optimizer, and loss function
input_shape = (3, 52, 64)  # Adjust as per your requirements
outputs_number = 262  # Number of output classes
lr = 0.00025  # Learning rate
model, optimizer, loss_function = get_pytorch_model(input_shape, outputs_number, lr)

# Train the model
train_model(model, train_loader, val_loader, optimizer, loss_function, epochs=10)
# Save the model
torch.save(model.state_dict(), 'fruit_classifier.pth')
