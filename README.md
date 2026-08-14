# F1 Tyre Degradation Predictor

Predicts how many laps a tyre stint can run before performance drops off, using 
real telemetry and weather data from the Belgian Grand Prix (2022–2025).

## Why this exists

Tyre strategy is one of the biggest levers a team has in a race — pit too early 
and you lose track position, pit too late and you're stuck losing time on a dead 
tyre. I wanted to see whether a simple model, using only publicly available 
telemetry, could estimate how long a given compound would realistically last 
under given conditions, rather than relying on the "typical" stint lengths teams 
quote in commentary.

## How it works

**1. Data**

Pulls official timing and weather data for Spa-Francorchamps (2022–2025) via the 
`FastF1` API.

**2. Isolating clean pace data**

Raw lap times are noisy — in-laps, out-laps, and laps behind a safety car don't 
reflect actual tyre performance. The script filters down to green-flag laps only, 
excluding in/out laps, so the pace trend reflects genuine tyre degradation rather 
than traffic or pit stop effects.

**3. Defining "drop-off"**

For each stint, laps are compared against the stint's early-lap baseline pace. 
The point where lap time increases by more than 1.03s is treated as the 
drop-off point — the target variable `LapsUntilDropoff` is the number of laps 
run before that happens.

**4. Model**

A scikit-learn pipeline:
- One-hot encoding for categorical features (`Driver`, `Compound`)
- Linear regression to predict `LapsUntilDropoff` from:
  - Driver
  - Tyre compound
  - Mean track temperature (°C)
  - Mean air temperature (°C)

Linear regression is a deliberately simple starting point — it's easy to inspect 
which features are actually driving the prediction, which matters more here than 
squeezing out extra accuracy with a black-box model.

## Running it

```bash
pip install fastf1 pandas scikit-learn
```

## Notes / caveats

- Single circuit (Spa) — degradation behavior is track-specific (abrasiveness, 
  layout, temperature swings), so this won't generalize to other tracks as-is.
- Driver is included as a feature, which captures some of the effect of driving 
  style on tyre wear, but also risks overfitting to specific drivers in the 
  dataset rather than learning general tyre behavior.
- Small sample size — only a handful of races per year at one circuit.
