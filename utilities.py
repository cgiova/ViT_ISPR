import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)


def permute(tensor):
    return tensor.permute(0, 3, 1,
                          2)  # Convert (batch_size, height, width, channels) to (batch_size, channels, height, width)


def preprocess(data):
    tensor = torch.from_numpy(data).float()
    tensor = permute(tensor)
    return tensor


class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = preprocess(np.load(data_path))  # Preprocess and convert data to tensor
        self.labels = torch.from_numpy(np.load(labels_path)).long()  # Load labels and convert to tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Function to create DataLoader
def create_data_loader(data_path, labels_path, batch_size, shuffle=True):
    dataset = CustomDataset(data_path, labels_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


## Model trainng and evaluation functions
def run_inference(dataloader, model, device):
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i + 1 <= len(dataloader):
                print(f'\nProcessing batch {i + 1} out of {len(dataloader)}')
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
            else:
                break

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    # Convert logits to probabilities
    probs = F.softmax(all_outputs, dim=1)

    # Get predicted classes
    _, preds = torch.max(probs, dim=1)

    print('Process finished')
    return all_outputs, all_labels, preds


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
