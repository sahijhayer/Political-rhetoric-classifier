import torch
from utils import Classifier
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

input_sentence = input("Enter your tweet\n")

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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

embedding = compute_embedding(input_sentence)

input_dim = embedding.shape[-1]
hidden_dim = 512
num_classes = 2
classifier = Classifier(input_dim, hidden_dim, num_classes)

classifier.load_state_dict(torch.load('model.pth'))
classifier.to(device)
classifier.eval()

embedding = torch.tensor(embedding, dtype=torch.float32)
embedding = embedding.to(device)

with torch.no_grad():
    outputs = classifier(embedding)
    prediction = torch.argmax(outputs, dim=0)
    probability = F.softmax(outputs, dim=0)
    print(probability)
