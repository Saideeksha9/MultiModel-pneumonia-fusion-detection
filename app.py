import os
import torch
import logging
import pandas as pd
from flask import Flask, render_template, redirect, url_for
from torchvision import transforms
from PIL import Image
from model import MultiModalModel
from dataset import extract_text

# ---------------------------------------------------------
# Silence Werkzeug/Flask HTTP Request Logs
# ---------------------------------------------------------
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset CSV
CSV_FILE = "chest_xray_multimodal_dataset.csv"
dataset_df = pd.read_csv(CSV_FILE)

# Load Vocab & Model State Dict
VOCAB = torch.load("vocab.pth", map_location=device)
state_dict = torch.load("model.pth", map_location=device)

# Auto-detect checkpoint vocabulary dimension
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

# Medical Information Mapping
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

def analyze_case(image_path, text):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    clean_text = extract_text(text) if "FINDINGS" in text or "IMPRESSION" in text else text.lower()
    tokens = [VOCAB.get(word, 0) for word in clean_text.split()]
    tokens = tokens[:100] + [0] * (100 - len(tokens))
    text_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # TEMPERATURE SCALER (Calibrates overconfident probability distributions)
    TEMPERATURE = 2.0  

    with torch.no_grad():
        outputs, _, _ = model(image_tensor, text_tensor)
        # Scaled softmax prevents single-class 99% overconfidence
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
        "probabilities": all_probabilities
    }

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('patient_case', index=0))

@app.route('/patient/<int:index>', methods=['GET'])
def patient_case(index):
    index = max(0, min(index, len(dataset_df) - 1))
    row = dataset_df.iloc[index]

    report_path = row['report_path']
    with open(report_path, 'r') as f:
        symptoms_text = f.read()

    image_path = row['image_path']
    result = analyze_case(image_path, symptoms_text)

    clean_symptoms_display = extract_text(symptoms_text)

    return render_template(
        'index.html',
        patient_index=index,
        total_patients=len(dataset_df),
        actual_label=row['diagnosis'],
        prediction=result["prediction"],
        confidence=result["confidence"],
        severity_level=result["medical"]["severity"],
        severity_color=result["medical"]["color"],
        symptoms=result["medical"]["symptoms"],
        suggestions=result["medical"]["suggestions"],
        precautions=result["medical"]["precautions"],
        input_symptoms=clean_symptoms_display,
        probabilities=result["probabilities"]
    )

if __name__ == '__main__':
    print("-------------------------------------------------------")
    print(" Starting Medical Diagnosis Dashboard")
    print(" Running at: http://127.0.0.1:5000")
    print("-------------------------------------------------------")
    app.run(debug=True, port=5000)