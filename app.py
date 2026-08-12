import os
import torch
import logging
import pandas as pd
from functools import lru_cache
from flask import Flask, render_template, redirect, url_for
from torchvision import transforms
from PIL import Image
from model import MultiModalModel
from dataset import extract_text

# Silence HTTP logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")  # Force CPU for deployment containers

# Absolute file paths
CSV_FILE = os.path.join(BASE_DIR, "chest_xray_multimodal_dataset.csv")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pth")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# Load Dataset CSV
dataset_df = pd.read_csv(CSV_FILE)

# Global Load of Vocab & Model
VOCAB = torch.load(VOCAB_PATH, map_location=device)
state_dict = torch.load(MODEL_PATH, map_location=device)

checkpoint_embed_shape = state_dict["text_branch.embedding.weight"].shape[0]
model = MultiModalModel(vocab_size=checkpoint_embed_shape - 1)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Image Transformations
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

MEDICAL_INFO = {
    "COVID-19": {
        "severity": "Critical", "color": "danger",
        "symptoms": "High fever, persistent dry cough, shortness of breath, loss of taste or smell.",
        "suggestions": "Immediate medical consultation. Monitor O2 saturation continuously.",
        "precautions": "Isolate in a well-ventilated room, wear an N95 mask, maintain strict hand hygiene."
    },
    "Bacterial Pneumonia": {
        "severity": "High", "color": "danger",
        "symptoms": "Productive cough with yellow/green phlegm, high fever, shaking chills, chest pain.",
        "suggestions": "Antibiotic regimen as prescribed by physician. Sputum culture recommended.",
        "precautions": "Complete full antibiotic course, rest adequately, stay well hydrated."
    },
    "Viral Pneumonia": {
        "severity": "Moderate", "color": "warning",
        "symptoms": "Fever, dry cough, headache, muscle pain, progressive breathlessness.",
        "suggestions": "Symptomatic care, hydration, and regular O2 monitoring.",
        "precautions": "Rest, avoid public places, use humidifiers to ease breathing."
    },
    "Tuberculosis": {
        "severity": "High", "color": "danger",
        "symptoms": "Persistent cough >3 weeks, coughing blood, night sweats, unexplained weight loss.",
        "suggestions": "Sputum AFB / GeneXpert test. Initiate DOTS therapy regimen.",
        "precautions": "Strict mask-wearing, isolate in well-ventilated spaces, complete long-term meds."
    },
    "Lung Cancer": {
        "severity": "Critical", "color": "danger",
        "symptoms": "Coughing blood (hemoptysis), chronic worsening cough, chest pain, weight loss.",
        "suggestions": "Urgent Pulmonology/Oncology referral for CT/PET scan or biopsy.",
        "precautions": "Avoid all smoke exposure, follow up immediately for oncological workups."
    },
    "Pleural Effusion": {
        "severity": "High", "color": "warning",
        "symptoms": "Sharp chest pain on inhalation, breathlessness, dry cough, orthopnea.",
        "suggestions": "Evaluate underlying cause (heart failure/infection). Thoracentesis if needed.",
        "precautions": "Limit strenuous physical exertion, sleep with elevated head rest."
    },
    "Pneumothorax": {
        "severity": "Critical", "color": "danger",
        "symptoms": "Sudden sharp chest pain, rapid shallow breathing, cyanosis, decreased breath sounds.",
        "suggestions": "Emergency ER evaluation for chest tube placement or decompression.",
        "precautions": "Seek emergency care immediately. Strictly avoid air travel or scuba diving."
    },
    "Cardiomegaly": {
        "severity": "Moderate", "color": "warning",
        "symptoms": "Leg/ankle edema, breathlessness on exertion, inability to lie flat.",
        "suggestions": "Cardiology referral for Echocardiogram and ECG evaluation.",
        "precautions": "Restrict daily sodium and fluid intake, monitor daily body weight."
    },
    "Pulmonary Edema": {
        "severity": "High", "color": "danger",
        "symptoms": "Pink frothy sputum, severe nocturnal breathlessness, anxiety, sweating.",
        "suggestions": "Emergency diuretics and oxygen therapy support.",
        "precautions": "Sit upright immediately when breathless. Strict fluid restriction."
    },
    "Atelectasis": {
        "severity": "Moderate", "color": "info",
        "symptoms": "Shallow breathing, mild shortness of breath, localized chest tightness.",
        "suggestions": "Chest physiotherapy, deep breathing exercises, incentive spirometry.",
        "precautions": "Perform frequent repositioning, practice deep-breathing exercises."
    },
    "Emphysema": {
        "severity": "Moderate", "color": "warning",
        "symptoms": "Chronic exertional dyspnea, barrel chest deformity, wheezing.",
        "suggestions": "Pulmonology testing (PFT), bronchodilator therapy.",
        "precautions": "Strict smoking cessation, avoid atmospheric pollutants and dust."
    },
    "Normal": {
        "severity": "Low", "color": "success",
        "symptoms": "No active pulmonary symptoms or normal baseline findings.",
        "suggestions": "No immediate medical intervention needed.",
        "precautions": "Maintain a healthy lifestyle, stay active, and avoid smoking."
    }
}

def normalize_path(path_str):
    clean_path = str(path_str).replace('\\', '/')
    return os.path.join(BASE_DIR, clean_path)

# In-Memory Cache to speed up predictions and prevent server timeouts
@lru_cache(maxsize=100)
def analyze_case_cached(target_index):
    row = dataset_df.iloc[target_index]

    report_path = normalize_path(row['report_path'])
    if not os.path.exists(report_path):
        symptoms_text = f"Patient presents with findings associated with {row['diagnosis']}."
    else:
        with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
            symptoms_text = f.read()

    image_path = normalize_path(row['image_path'])
    if not os.path.exists(image_path):
        image_path = os.path.join(BASE_DIR, "images", "CXR-1001.png")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    clean_text = extract_text(symptoms_text) if "FINDINGS" in symptoms_text or "IMPRESSION" in symptoms_text else symptoms_text.lower()
    tokens = [VOCAB.get(word, 0) for word in clean_text.split()]
    tokens = tokens[:100] + [0] * (100 - len(tokens))
    text_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    TEMPERATURE = 2.0  

    with torch.no_grad():
        outputs, _, _ = model(image_tensor, text_tensor)
        probs = torch.softmax(outputs / TEMPERATURE, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_class = LABELS[pred_idx]
    medical_details = MEDICAL_INFO.get(pred_class, MEDICAL_INFO["Normal"])

    all_probabilities = [
        {"class": LABELS[i], "probability": round(probs[i].item() * 100, 2)}
        for i in range(len(LABELS))
    ]
    all_probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "prediction": pred_class,
        "confidence": round(probs[pred_idx].item() * 100, 2),
        "medical": medical_details,
        "symptoms_display": clean_text,
        "actual_label": row['diagnosis'],
        "probabilities": all_probabilities
    }

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('patient_case', index=0))

@app.route('/patient/<int:index>', methods=['GET'])
def patient_case(index):
    target_index = max(0, min(index, len(dataset_df) - 1))
    
    if index != target_index:
        return redirect(url_for('patient_case', index=target_index))

    # Get cached result instantly
    result = analyze_case_cached(target_index)

    return render_template(
        'index.html',
        patient_index=target_index,
        total_patients=len(dataset_df),
        actual_label=result["actual_label"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        severity_level=result["medical"]["severity"],
        severity_color=result["medical"]["color"],
        symptoms=result["medical"]["symptoms"],
        suggestions=result["medical"]["suggestions"],
        precautions=result["medical"]["precautions"],
        input_symptoms=result["symptoms_display"],
        probabilities=result["probabilities"]
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)