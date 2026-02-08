# GenTwin: GenAI-Driven Cyber-Physical Intrusion Detection System for Smart Water Infrastructure

## Abstract

GenTwin is an advanced cybersecurity system designed to protect industrial water treatment infrastructure from cyber-physical attacks. It combines deep learning anomaly detection, large language model reasoning, and digital twin simulation to autonomously discover vulnerabilities, generate realistic adversarial attacks, detect anomalies, and explain security risks.

The system demonstrates how artificial intelligence can proactively secure critical infrastructure by simulating and detecting threats before real-world impact occurs.

---

## System Demonstration

### Reconstruction Error Distribution

This chart shows how the Variational Autoencoder distinguishes normal and attack behavior.

![Error Distribution](backend/pics/legitimacy_chart1_error_distribution.png)

---

### ROC Curve (Detection Performance)

The model achieves strong separation between normal and anomalous system behavior.

![ROC Curve](backend/pics/legitimacy_chart2_roc_curve.png)

---

### Confusion Matrix

Shows accurate classification of attack versus normal samples.

![Confusion Matrix](backend/pics/legitimacy_chart3_confusion_matrix.png)

---

### Detection Severity Breakdown

Visualizes attack severity levels detected by the system.

![Severity Breakdown](backend/pics/legitimacy_chart4_severity_breakdown.png)

---

### Detection Timeline

Shows how anomalies emerge and are detected over time.

![Detection Timeline](backend/pics/legitimacy_chart5_detection_timeline.png)

---

## Key Capabilities

### AI-Based Anomaly Detection

* Variational Autoencoder trained on normal system behavior
* Detects anomalies using reconstruction error
* Adaptive threshold selection
* High sensitivity to cyber-physical deviations

---

### GenAI Vulnerability Discovery

Powered by Llama-3.1 via Groq API:

* Automatically discovers cybersecurity weaknesses
* Identifies vulnerable sensors and actuators
* Generates realistic adversarial attack scenarios
* Provides reasoning-based explanations

---

### Digital Twin Simulation

* Simulates water treatment plant processes
* Validates physical plausibility of attacks
* Enables safe cybersecurity testing environment
* Prevents real-world operational risk

---

### Explainable Security Analysis

* Explains detection outcomes
* Identifies root causes of anomalies
* Recommends mitigation strategies
* Provides operator-friendly insights

---

### Interactive Monitoring Dashboard

Frontend visualization includes:

* Live anomaly detection display
* Attack simulation controls
* Sensor and actuator visualization
* System health monitoring

---

## Architecture Overview

```
              GenAI (Llama-3.1)
                     │
                     ▼
        Vulnerability Discovery Engine
                     │
                     ▼
           Adversarial Attack Generator
                     │
                     ▼
           Digital Twin Simulation Layer
                     │
                     ▼
       Variational Autoencoder Detector
                     │
                     ▼
         Detection and Risk Analysis
                     │
                     ▼
            Visualization Dashboard
```

---

## Project Structure

```
IITK_CR2/
│
├ backend/
│   ├ app_gentwin.py
│   ├ enhanced_detection_system.py
│   ├ genai_vulnerability_scanner.py
│   ├ digital_twin.py
│   ├ vulnerability_db.py
│   └ pics/
│
├ iik-frontend-/
│   ├ src/
│   ├ components/
│   └ package.json
│
├ models/
│   ├ train_vae.py
│   ├ evaluate_vae.py
│   └ vae_model.pth
│
├ preprocess/
│   ├ clean_data.py
│   └ eda.py
│
└ README.md
```

---

## Installation Guide

### Backend Setup

```
cd backend
pip install -r requirements.txt
```

Set Groq API key:

Windows:

```
setx GROQ_API_KEY "your_api_key"
```

Linux / Mac:

```
export GROQ_API_KEY="your_api_key"
```

Run backend:

```
python app_gentwin.py
```

Backend runs at:

```
http://127.0.0.1:5000
```

---

### Frontend Setup

```
cd iik-frontend-
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## Detection Methodology

The Variational Autoencoder learns normal system patterns.

Detection logic:

```
reconstruction_error = model(sensor_data)

if reconstruction_error > threshold:
    anomaly_detected = True
else:
    normal_operation
```

---

## GenAI Security Pipeline

The LLM performs:

1. System analysis
2. Vulnerability identification
3. Attack generation
4. Detection evasion analysis
5. Security recommendations

---

## Technology Stack

Backend:

* Python
* PyTorch
* Flask
* NumPy

Frontend:

* React
* Vite
* JavaScript

AI and GenAI:

* Variational Autoencoder
* Llama-3.1 LLM
* Groq API

Visualization:

* Matplotlib
* Chart rendering

---

## Use Cases

This system applies to:

* Water treatment plants
* Industrial control systems
* Critical infrastructure security
* Smart city protection
* Cyber-physical systems

---

## Research Contribution

GenTwin demonstrates how generative AI and deep learning can be combined to create autonomous cybersecurity defense systems capable of:

* Discovering vulnerabilities
* Simulating attacks
* Detecting anomalies
* Explaining security risks

---

## Hackathon Submission

Hack IITK 2026
GenAI Cybersecurity Track

---

## Author

Giri YZ
B.Tech Computer Science and Engineering
Cybersecurity and IoT Specialization

Sri Ramachandra Institute of Higher Education and Research

---

## License

MIT License
