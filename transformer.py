import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch import device
from tqdm import tqdm
import csv

def parse_trump_data(filename):
    with open(filename, mode="r", encoding='utf-8') as f:
        reader = csv.reader(f)
        texts = []
        for line in reader:
            if "RT" not in line[1]:
                texts.append(line[1])
        return np.array(texts)

def parse_biden_data(filename):
    with open(filename, mode="r", encoding='utf-8') as f:
        reader = csv.reader(f)
        texts = []
        for line in reader:
            texts.append(line[0])
        return np.array(texts)

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
batch_size = 128

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=280
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

trump_data = parse_trump_data('trump_tweets.csv')
biden_data = parse_biden_data('biden_tweets.csv')

n = len(biden_data)

data = np.concatenate((np.random.choice(trump_data, n, replace=False), biden_data))
labels = np.concatenate((np.zeros(n), np.ones(n)))

combined = np.stack((data, labels), axis=1)

np.random.shuffle(combined)

data = combined[:, 0]
labels = combined[:, 1]

embeddings = []
for text in tqdm(data, desc="Computing embeddings"):
    embedding = compute_embedding(text, tokenizer, model)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

np.save('embeddings.npy', embeddings)
np.save('labels.npy', labels)