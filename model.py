import numpy as np
import torch
import torch.nn as nn
from torch import device
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import Classifier, EmbeddingDataset

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_loaders(dataset,
                  train_ratio: float=0.85,
                  val_ratio: float=0.15,
                  batch_size:int=batch_size):
    train_size = int(train_ratio * len(dataset)) + 1
    val_size = int(val_ratio * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

@torch.no_grad
def eval_accuracy(model, eval_loader, device):
    model.eval()
    n_correct = 0
    n_total = 0
    for embeddings, labels in eval_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        logits = model(embeddings)
        _, pred = torch.max(logits, 1)
        n_correct += (pred == labels).sum().item()
        n_total += len(labels)
    return n_correct / n_total

embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')

embeddings = embeddings.astype(float)
labels = labels.astype(float)

input_dim = embeddings.shape[-1]
hidden_dim = 512
num_classes = 2
classifier = Classifier(input_dim, hidden_dim, num_classes).to(device)

dataset = EmbeddingDataset(embeddings, labels)
train_loader, val_loader = create_loaders(dataset)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

num_epochs = 150
train_error_rates, val_error_rates = [], []
running_loss, running_steps = 0.0, 0
eval_interval = 10

for epoch in tqdm(range(1, num_epochs + 1)):
    classifier.train()
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_steps += 1
    if epoch % eval_interval == 0:
        train_error = 1 - eval_accuracy(classifier, train_loader, device)
        val_error = 1 - eval_accuracy(classifier, val_loader, device)
        val_error_rates.append(val_error)
        train_error_rates.append(train_error)
        print('Epoch: %d' % epoch)
        print('Average training loss: %.3f' % (running_loss / running_steps))
        print('Current validation error rate: %.3f' % (val_error * 100.0))
        running_loss, running_steps = 0., 0

model_path = "model.pth"

torch.save(classifier.state_dict(), model_path)