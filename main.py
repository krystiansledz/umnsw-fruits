import os
import time

import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.quantization
import copy

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# torch.backends.quantized.engine = 'fbgemm'
# torch.backends.quantized.engine = 'qnnpack' # for mac
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # for mac

# device = torch.device("cpu")
print(f"Using device: {device}.")

input_shape = (3, 52, 64)  # Adjust as per your requirements
outputs_number = 262  # Number of output classes
# Initialize the model, optimizer, and loss function
lr = 0.00025  # Learning rate


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
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
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


class ThirdModel(nn.Module):
    def __init__(self, input_shape, outputs_number):
        super(ThirdModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self._to_linear = None
        self.convs(torch.zeros(1, *input_shape))

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, outputs_number)

    def convs(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
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
    model = ThirdModel(input_shape, outputs_number).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    return model, optimizer, loss_function

model, optimizer, loss_function = get_pytorch_model(input_shape, outputs_number, lr)

def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return round(100 * correct / total, 2)


def top_n_accuracy(model, data_loader, n=3):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, top3_indices = torch.topk(outputs, n, dim=1)
            top3_indices = top3_indices.cpu().numpy()
            labels = labels.cpu().numpy()

            for idx, label in enumerate(labels):
                if label in top3_indices[idx]:
                    correct += 1
            total += labels.shape[0]


    return round(100 * correct / total, 2)

def evaluate_or_train_loss(model, data_loader, loss_function, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(train):
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    return round(100 * total_loss / total_count, 2)

def train_model(model, train_loader, val_loader, optimizer, loss_function, epochs=10, model_path='model_epoch_'):
    model = model.to(device)

    for epoch in range(epochs):
        start_time = time.perf_counter()

        # Trening modelu i obliczanie średniej straty na zbiorze treningowym
        train_loss = evaluate_or_train_loss(model, train_loader, loss_function, optimizer, train=True)

        # Ewaluacja modelu i obliczanie średniej straty na zbiorze walidacyjnym
        val_loss = evaluate_or_train_loss(model, val_loader, loss_function, train=False)
        accuracy = evaluate_accuracy(model, val_loader)

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} took {round(duration, 2)}s, Training Loss: {round(train_loss, 2)}, Validation Loss: {round(val_loss, 2)}, Accuracy: {accuracy}%")

        # Zapisanie modelu po każdej epoce
        epoch_model_path = f"models/{model_path}-{epoch + 1}.pth"
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

    # Etykiety dla każdego obrazu
    labels = [label for _, label in dataset.imgs]

    # Utworzenie instancji StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size / len(dataset), random_state=0)

    # Podział na indeksy treningowe i walidacyjne
    train_indices, val_indices = next(sss.split(np.zeros(len(labels)), labels))

    # Utworzenie podzbiorów
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_loader_with_reduced_batch_size = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=False)

    return train_loader, val_loader, classes, val_loader_with_reduced_batch_size

def start_training(model_name):
    # Train the model
    start_time = time.perf_counter()
    print("Start training.")
    train_model(model, train_loader, val_loader, optimizer, loss_function, epochs=250, model_path=model_name)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"The training took {duration} seconds to execute.")
    # Save the model
    torch.save(model.state_dict(), f'models/{model_name}.pth')

#### LOAD MODEL ####
def load_model(path):
    loaded_model = ThirdModel(input_shape, outputs_number)
    loaded_model.load_state_dict(torch.load(path, map_location=device))
    loaded_model = loaded_model.to(device)
    return loaded_model

def analyze_model(model, train_loader, val_loader, name=""):
    # Ocena dokładności i straty
    train_accuracy = evaluate_accuracy(model, train_loader)
    val_accuracy = evaluate_accuracy(model, val_loader)
    val_loss = evaluate_or_train_loss(model, val_loader, loss_function)

    print(
        f"{name} Train Acc: {train_accuracy}, Val Acc: {val_accuracy}, Val Loss: {val_loss}")
def conf_matrix(loaded_model, val_loader):
    def get_all_preds(model, loader):
        all_preds = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds = torch.cat((all_preds, preds), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
        return all_preds.cpu().numpy(), all_labels.cpu().numpy()

    # Assuming val_loader is defined and loaded with the validation dataset
    predictions, labels = get_all_preds(loaded_model, val_loader)

    # Compute the confusion matrix
    return confusion_matrix(labels, predictions)

# Function to identify most confused labels
def find_most_confused_labels(loaded_model, val_loader, classes):
    conf_mat = conf_matrix(loaded_model, val_loader)
    # Zero out the diagonal elements
    np.fill_diagonal(conf_mat, 0)

    # Find the indices of the maximum values
    max_indices = np.argwhere(conf_mat == np.max(conf_mat))

    # Map indices to class labels
    confused_labels = [(classes[i], classes[j]) for i, j in max_indices]

    return confused_labels

#### OPTYMALIZACJA SIECI ####
def apply_pruning(model):
    # Przycinanie warstw konwolucyjnych
    prune.l1_unstructured(model.conv1, name='weight', amount=0.1)
    prune.l1_unstructured(model.conv2, name='weight', amount=0.15)
    prune.l1_unstructured(model.conv3, name='weight', amount=0.2)

    # Przycinanie warstw liniowych
    prune.l1_unstructured(model.fc1, name='weight', amount=0.1)
    prune.l1_unstructured(model.fc2, name='weight', amount=0.1)

    # Make pruning permanent
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    return model


def apply_quantization(model, data_loader):
    model.to("cpu").eval()

    # Ustawienie konfiguracji kwantyzacji
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # model.qconfig = torch.quantization.get_default_qconfig('qnnpack') # for mac

    # Przygotowanie modelu do kwantyzacji
    torch.quantization.prepare(model, inplace=True)

    # Kalibracja modelu
    with torch.no_grad():
        for inputs, _ in data_loader:
            model(inputs.to('cpu'))

    # Konwersja modelu do kwantyzowanej wersji
    torch.quantization.convert(model, inplace=True)

    return model

def evaluate_inference_time(model, data_loader, name=""):
    model.eval()
    total_time = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            total_count += inputs.size(0)

    average_inference_time = total_time / total_count
    print(f"Average inference time for model {name}: {round(average_inference_time, 6)} s/record")

    return average_inference_time

train_loader, val_loader, classes, val_loader_with_reduced_batch_size = get_dataset()

# start_training("model3v2")

model_name = "model3v2"
loaded_model = load_model(f"models/{model_name}.pth")

# Assuming conf_mat is your confusion matrix and classes is a list of class names
most_confused = find_most_confused_labels(loaded_model, val_loader, classes)

print("Most confused label pairs:", most_confused)

# Ocena dokładności pierwotnego modelu
# analyze_model(model=loaded_model, train_loader=train_loader, val_loader=val_loader, name="Original")
# evaluate_inference_time(loaded_model, val_loader, name="Original")
top_3_accuracy = top_n_accuracy(loaded_model, val_loader, 3)
top_4_accuracy = top_n_accuracy(loaded_model, val_loader, 4)
top_5_accuracy = top_n_accuracy(loaded_model, val_loader, 5)
print(f"Top-3 Accuracy: {top_3_accuracy}")
print(f"Top-4 Accuracy: {top_4_accuracy}")
print(f"Top-5 Accuracy: {top_5_accuracy}")

# Pruning
# model_pruned = copy.deepcopy(loaded_model)
# apply_pruning(model_pruned)
# analyze_model(model=model_pruned, train_loader=train_loader, val_loader=val_loader, name="Pruned")
# # evaluate_inference_time(model_pruned, val_loader, name="Pruned")
# torch.save(model_pruned.state_dict(), f"models/{model_name}-pruned2.pth")

# # Kwantyzacja
# model_quantized = copy.deepcopy(loaded_model)
# apply_quantization(model_quantized, val_loader_with_reduced_batch_size)
# analyze_model(model=model_quantized, train_loader=train_loader, val_loader=val_loader, name="Quantized")
# evaluate_inference_time(model_quantized, val_loader, name="Quantized")
# torch.save(model_pruned.state_dict(), f"models/{model_name}-quantized.pth")

#
# # Pruning i Kwantyzacja
# model_pruned_quantized = copy.deepcopy(loaded_model)
# apply_pruning(model_pruned_quantized)
# apply_quantization(model_pruned_quantized, val_loader)
# pruned_quantized_accuracy = evaluate_accuracy(model_pruned_quantized, val_loader)
# print(f"Pruned and Quantized Model Accuracy: {pruned_quantized_accuracy}%")
