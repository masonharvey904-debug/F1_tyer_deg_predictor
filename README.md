# Formula 1 Tyre Degradation & Drop-off Prediction

A machine learning project using `FastF1` telemetry and weather data from the Belgian Grand Prix (2022–2025) to predict tyre performance drop-off laps based on compound and track conditions.

---

## 🏎️ Overview

* **Data Source:** Official F1 timing and weather telemetry via the `FastF1` API.
* **Methodology:** Filters accurate green-flag stints, calculates pace decay relative to best stint pace, and identifies thermal/wear drop-off thresholds.
* **Model:** Scikit-Learn pipeline featuring One-Hot Encoding for categorical features (`Driver`, `Compound`) and Linear Regression for prediction.

---

## 📊 Features & Target

* **Features:** Driver, Tyre Compound, Mean Track Temperature (°C), Mean Air Temperature (°C).
* **Target Variable:** `LapsUntilDropoff` (Number of laps before stint pace degrades by > 1.03s).

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install fastf1 pandas scikit-learn
