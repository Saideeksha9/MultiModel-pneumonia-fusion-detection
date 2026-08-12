import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# -------- IMAGE MODEL --------
class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Uses modern weights API to avoid deprecation warnings
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 1024)

    def forward(self, x):
        return self.cnn(x)

# -------- TEXT MODEL --------
class TextBranch(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256):
        super().__init__()
        # Accommodate padding index 0 correctly
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 256, batch_first=True)
        self.fc = nn.Linear(256, 768)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# -------- FUSION --------
class FusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wi = nn.Linear(1024, 512)
        self.Wt = nn.Linear(768, 512)
        self.w = nn.Linear(512, 1)

    def forward(self, fI, fT):
        hI = self.Wi(fI)
        hT = self.Wt(fT)

        aI = torch.exp(self.w(torch.tanh(hI)))
        aT = torch.exp(self.w(torch.tanh(hT)))

        alpha_I = aI / (aI + aT)
        alpha_T = aT / (aI + aT)

        fused = alpha_I * hI + alpha_T * hT
        return fused, hI, hT

# -------- FINAL MODEL --------
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size=5000):
        super().__init__()
        self.image_branch = ImageBranch()
        self.text_branch = TextBranch(vocab_size=vocab_size)
        self.fusion = FusionLayer()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 12)
        )

    def forward(self, image, text):
        fI = self.image_branch(image)
        fT = self.text_branch(text)

        fused, hI, hT = self.fusion(fI, fT)
        output = self.classifier(fused)

        return output, hI, hT