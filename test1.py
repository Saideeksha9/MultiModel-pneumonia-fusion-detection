import torch
from torchvision import transforms
from PIL import Image
from model import MultiModalModel
from dataset import extract_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = torch.load("vocab.pth")

# Load model passing saved vocabulary length
model = MultiModalModel(vocab_size=len(vocab))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Added ImageNet Normalization matching training pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABELS = [
    "Normal", "Bacterial Pneumonia", "Viral Pneumonia",
    "COVID-19", "Tuberculosis", "Lung Cancer",
    "Pleural Effusion", "Pneumothorax", "Cardiomegaly",
    "Pulmonary Edema", "Atelectasis", "Emphysema"
]

def predict(image_path, text):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Process input text consistent with dataset processing
    clean_text = extract_text(text) if "FINDINGS" in text or "IMPRESSION" in text else text.lower()
    
    tokens = [vocab.get(word, 0) for word in clean_text.split()]
    tokens = tokens[:100] + [0] * (100 - len(tokens))
    text_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _ = model(image, text_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs

if __name__ == "__main__":
    pred, probs = predict("images/CXR-1004.png", "patient has cough and fever")

    print(f"Prediction: {LABELS[pred]}")
    print(f"Confidence: {probs[0][pred].item() * 100:.2f}%")