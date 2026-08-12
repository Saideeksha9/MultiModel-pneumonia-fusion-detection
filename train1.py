print("File started")
print("Importing torch...")
import torch
print("Importing torchvision...")

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

print("Importing dataset...")
from dataset import MultiModalDataset
print("Importing model...")
from model import MultiModalModel
print("All imports done ✅")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization standard
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset")
dataset = MultiModalDataset(
    "chest_xray_multimodal_dataset.csv",
    "images",
    "reports",
    transform
)
print("Dataset loaded")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Loading model")
model = MultiModalModel(vocab_size=len(dataset.vocab)).to(device)
print("Model Loaded")

# Label smoothing to prevent 100% overconfidence
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Weight decay to prevent overfitting on small datasets
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

def consistency_loss(pI, pT):
    return torch.mean((pI - pT) ** 2)

for epoch in range(25):
    model.train()
    total_loss = 0

    for img, text, labels in loader:
        img = img.to(device)
        text = text.to(device)
        labels = labels.to(device)

        outputs, hI, hT = model(img, text)

        loss_ce = criterion(outputs, labels)

        pI = torch.softmax(hI, dim=1)
        pT = torch.softmax(hT, dim=1)

        loss_cons = consistency_loss(pI, pT)

        loss = loss_ce + 0.1 * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
torch.save(dataset.vocab, "vocab.pth")

print("Training complete!")