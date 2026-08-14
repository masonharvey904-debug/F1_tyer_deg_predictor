# Formula 1 Tyre Degradation & Drop-off Prediction

A machine learning project using `FastF1` telemetry and weather data from the Belgian Grand Prix (2022–2025) to predict stint length based and track conditions and tyer compound.



##  Overview

* **Data Source:** Official F1 timing and weather telemetry from the `FastF1` API.
* **Methodology:** Filters for green flag laps which are not out or in laps to identify the amount of laps before a performance drop off occurs.  
* **Model:** Scikit-Learn pipeline featuring One-Hot Encoding for categorical features (`Driver`, `Compound`) and Linear Regression for prediction.



##  Features & Target

* **Features:** Driver, Tyre Compound, Mean Track Temperature (°C), Mean Air Temperature (°C).
* **Target Variable:** `LapsUntilDropoff` (Number of laps before stint pace degrades by > 1.03s).



##  How to Run

1. Install dependencies:
   ```bash
   pip install fastf1 pandas scikit-learn
