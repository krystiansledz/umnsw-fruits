import time

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}.")


class FirstModel(nn.Module):
    def __init__(self, input_shape, outputs_number):
        super(FirstModel, self).__init__()

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


class SecondModel(nn.Module):
    def __init__(self, input_shape, outputs_number):
        super(SecondModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened features after the conv and pool layers
        self._to_linear = None
        self.convs(torch.zeros(1, *input_shape))

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, outputs_number)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        if self._to_linear is None:
            self._to_linear = x[0].numel()
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_pytorch_model(input_shape, outputs_number, lr):
    model = SecondModel(input_shape, outputs_number).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    return model, optimizer, loss_function


def train_model(model, train_loader, val_loader, optimizer, loss_function, epochs=10, model_path='model_epoch_'):
    model = model.to(device)

    for epoch in range(epochs):
        start_time = time.perf_counter()

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} took {round(duration, 2)}s, Loss: {round(running_loss / len(train_loader), 2)} Accuracy: {round(100 * correct / total, 2)}%")

        # Save the model after each epoch
        epoch_model_path = f"models/{model_path}{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_model_path)


def get_dataset():
    # Training and validation loop
    transform = transforms.Compose([
        transforms.Resize((52, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root='52x64', transform=transform)
    classes = dataset.classes

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, classes


train_loader, val_loader, classes = get_dataset()

# Initialize the model, optimizer, and loss function
input_shape = (3, 52, 64)  # Adjust as per your requirements
outputs_number = 262  # Number of output classes
lr = 0.00025  # Learning rate
model, optimizer, loss_function = get_pytorch_model(input_shape, outputs_number, lr)

# Train the model
start_time = time.perf_counter()
print("Start training.")
train_model(model, train_loader, val_loader, optimizer, loss_function, epochs=10, model_path='model_gpu')
end_time = time.perf_counter()
duration = end_time - start_time
print(f"The training took {duration} seconds to execute.")
# Save the model
torch.save(model.state_dict(), 'models/model_gpu.pth')

# loaded_model = CustomPyTorchModel(input_shape, outputs_number)

# # Load the saved state dictionary
# loaded_model.load_state_dict(torch.load('fruit_classifier.pth'))
#
# print("Model loaded successfully.")
#
# def get_all_preds(model, loader):
#     all_preds = torch.tensor([]).to('cpu')
#     all_labels = torch.tensor([]).to('cpu')
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to('cpu'), labels.to('cpu')
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds = torch.cat((all_preds, preds), dim=0)
#             all_labels = torch.cat((all_labels, labels), dim=0)
#     return all_preds.cpu().numpy(), all_labels.cpu().numpy()
#
# # Assuming val_loader is defined and loaded with the validation dataset
# start_time = time.perf_counter()
# predictions, labels = get_all_preds(loaded_model, val_loader)
# end_time = time.perf_counter()
# duration = end_time - start_time
# print(f"The function took {duration} seconds to execute.")
#
# # Compute the confusion matrix
# conf_mat = confusion_matrix(labels, predictions)
#
# # Plot the confusion matrix
# plt.figure(figsize=(10,10))
# sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
#
# # Function to identify most confused labels
# def find_most_confused_labels(conf_mat, classes):
#     # Zero out the diagonal elements
#     np.fill_diagonal(conf_mat, 0)
#
#     # Find the indices of the maximum values
#     max_indices = np.argwhere(conf_mat == np.max(conf_mat))
#
#     # Map indices to class labels
#     confused_labels = [(classes[i], classes[j]) for i, j in max_indices]
#
#     return confused_labels
#
# # Assuming conf_mat is your confusion matrix and classes is a list of class names
# most_confused = find_most_confused_labels(conf_mat, classes)
#
# print("Most confused label pairs:", most_confused)
