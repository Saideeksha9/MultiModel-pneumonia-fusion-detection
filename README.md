# 🩺 Multimodal Pneumonia Detection

A deep learning-based system for pneumonia detection using **Chest X-Ray images and clinical reports**. The project combines visual and textual information to improve pneumonia classification.

 🚀 Live Demo
     web-production-29fe0.up.railway.app
 
## 🚀 Features

- 🩻 Chest X-Ray image analysis
- 📝 Clinical report analysis
- 🧠 Multimodal Deep Learning
- 🔗 Image and text feature fusion
- 📊 Pneumonia classification
- 📈 Prediction results and evaluation

## 🛠️ Tech Stack

- Python
- PyTorch
- TorchVision
- ResNet18
- LSTM
- Pandas
- NumPy
- PIL

Example output may include:

| Condition | Prediction |
|-----------|------------|
| Normal | 73.32% |
| Emphysema | 3.58% |
| Pneumothorax | 3.08% |
| Atelectasis | 2.72% |

## 🔄 Workflow

```text
Chest X-Ray Image ──→ ResNet18 ──→ Image Features ──┐
                                                     │
                                                     ├──→ Feature Fusion
                                                     │         ↓
Clinical Report ──→ LSTM ──→ Text Features ─────────┘
                                                               ↓
                                                        Classification
                                                               ↓
                                                   Pneumonia Prediction
