import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from collections import Counter

# -------- TEXT CLEANING --------
def extract_text(report):
    sections = ["CLINICAL INDICATION", "FINDINGS", "IMPRESSION"]
    text = ""

    for sec in sections:
        if sec in report:
            part = report.split(sec)[1]
            text += part.split("\n\n")[0] + " "

    return text.lower()

# -------- BUILD VOCAB --------
def build_vocab(texts, max_size=5000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: i+1 for i, (word, _) in enumerate(counter.most_common(max_size))}
    return vocab

def text_to_indices(text, vocab, max_len=100):
    tokens = [vocab.get(word, 0) for word in text.split()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens)

# -------- DATASET --------
class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, report_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.transform = transform

        # Load all reports for vocab
        texts = []
        for _, row in self.data.iterrows():
            report_path = row['report_path']
            with open(report_path, "r") as f:
                report = f.read()
                texts.append(extract_text(report))

        self.vocab = build_vocab(texts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        report_path = row['report_path']
        with open(report_path, "r") as f:
            report = f.read()
        text = extract_text(report)
        text_indices = text_to_indices(text, self.vocab)
        label = torch.tensor(row['label'])
        return image, text_indices, label