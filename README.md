# Predictive Maintenance System for Power Electronics (Digital Twin)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-Prototype-green)

A portfolio-grade, end-to-end predictive maintenance system designed for DC-DC converters. 
This project demonstrates **Systems Thinking** by combining behavioral physics simulation with industrial machine learning to detect early-stage degradation (Capacitor Aging, MOSFET Wear) before catastrophic failure.

## 🎯 Engineering Goal
To replace traditional threshold-based monitoring with a data-driven approach that can:
1.  **Detect Anomalies** in multivariate sensor streams (Voltage, Current, Temperature, Ripple).
2.  **Classify Faults** specifically (e.g., distinguishing between Thermal Runaway vs. Capacitor drying).
3.  **Predict Remaining Useful Life (RUL)**: Estimating how many operational hours are left before failure.
4.  **Operate in Real-Time** using a streaming architecture.

## 🏗 Architecture

### 1. The Physics Engine (Digital Twin)
Located in `src/simulation/`, the `BuckConverterSimulator` implements a differential-equation-based model of a Step-Down Converter. 
-   **Features**: Dynamic load, switching ripple approximation, thermal modeling ($P_{loss} \to \Delta T$).
-   **Fault Injection**: Programmatic control over $ESR$, $C$, $R_{ds(on)}$, and $R_{thermal}$ to simulate aging.

### 2. The ML Pipeline
Located in `src/pipeline/` and `src/models/`.
-   **Feature Engineering**: Rolling window statistics (Mean, Std, Skew) + Frequency-domain proxies (Ripple magnitude).
-   **Stage 1: Anomaly Detection**: `IsolationForest` trained *only* on healthy data.
-   **Stage 2: Fault Classification**: `XGBoost` Classifier for root-cause analysis.
-   **Stage 3: Prognostics (RUL)**: `RandomForestRegressor` trained on run-to-failure simulations to predict remaining life cycles.

### 3. Dashboard
Located in `src/dashboard/`.
-   A **Streamlit** application serving as the HMI (Human Machine Interface).
-   Allows real-time "Health Status" monitoring and manual fault injection to test the AI's response.
-   **RUL Gauge**: Displays estimated remaining life in real-time.

## 🚀 Getting Started

### Prerequisites
-   Python 3.9+
-   Command Line Interface

### Installation
```bash
git clone <your-repo-url>
cd power-guard-ai
pip install -r requirements.txt
```

### Running the System
1.  **Generate Data & Train Models**:
    ```bash
    # Generate Physics Data (Healthy, Faulty, and Run-to-Failure)
    python scripts/generate_datasets.py
    python scripts/generate_rul_data.py
    
    # Train Models
    python src/models/train.py
    python src/models/train_rul.py
    ```

2.  **Launch the Dashboard**:
    ```bash
    streamlit run src/dashboard/app.py
    ```

## 📂 Project Structure
```
├── data/                   # Generated CSV datasets (ignored by git)
├── scripts/                # Utility scripts (Data Gen)
├── src/
│   ├── dashboard/          # Streamlit App
│   ├── models/             # ML Training & Inference Logic
│   ├── pipeline/           # Feature Engineering
│   ├── simulation/         # Physics Engine
│   └── utils/              # Visualization helpers
├── tests/                  # Verification scripts
├── requirements.txt
└── README.md
```

## 👨‍💻 Author
Designed as a demonstration of intersectional skills in **Electrical Engineering** and **Machine Learning**.
