import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIGURATION
# ==============================

NORMAL_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\normal.csv"
ATTACK_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\attack.csv"

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("SWaT DATASET EDA STARTED")
print("="*60)

# ==============================
# LOAD DATA
# ==============================

print("\nLoading datasets...")

normal_df = pd.read_csv(NORMAL_PATH)
attack_df = pd.read_csv(ATTACK_PATH)

print(f"\nNormal dataset shape: {normal_df.shape}")
print(f"Attack dataset shape: {attack_df.shape}")

# ==============================
# BASIC INFO
# ==============================

print("\nColumn names:")
print(normal_df.columns.tolist())

print("\nData types:")
print(normal_df.dtypes)

# ==============================
# MISSING VALUES
# ==============================

print("\nChecking missing values...")

missing = normal_df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

# ==============================
# STATISTICAL SUMMARY
# ==============================

print("\nStatistical summary:")

stats = normal_df.describe()
stats.to_csv(os.path.join(OUTPUT_DIR, "statistical_summary.csv"))

print(stats)

# ==============================
# LABEL DISTRIBUTION
# ==============================

print("\nLabel distribution:")

print("Normal dataset:")
print(normal_df["Normal/Attack"].value_counts())

print("\nAttack dataset:")
print(attack_df["Normal/Attack"].value_counts())

# ==============================
# SENSOR VS ACTUATOR DETECTION
# ==============================

print("\nClassifying columns...")

sensor_cols = []
actuator_cols = []

for col in normal_df.columns:
    if col.startswith("LIT") or col.startswith("FIT") or col.startswith("AIT") or col.startswith("PIT") or col.startswith("DPIT"):
        sensor_cols.append(col)
    elif col.startswith("MV") or col.startswith("P") or col.startswith("UV"):
        actuator_cols.append(col)

print(f"\nTotal sensors: {len(sensor_cols)}")
print(sensor_cols)

print(f"\nTotal actuators: {len(actuator_cols)}")
print(actuator_cols)

# ==============================
# ACTUATOR UNIQUE VALUES
# ==============================

print("\nActuator value analysis:")

actuator_analysis = {}

for col in actuator_cols:
    unique_vals = normal_df[col].unique()
    actuator_analysis[col] = unique_vals

    print(f"{col}: {unique_vals}")

# Save actuator analysis
pd.DataFrame(dict([(k,pd.Series(v)) for k,v in actuator_analysis.items()])).to_csv(
    os.path.join(OUTPUT_DIR, "actuator_values.csv")
)

# ==============================
# SENSOR RANGE ANALYSIS
# ==============================

print("\nSensor range analysis:")

sensor_ranges = []

for col in sensor_cols:
    sensor_ranges.append({
        "Sensor": col,
        "Min": normal_df[col].min(),
        "Max": normal_df[col].max(),
        "Mean": normal_df[col].mean(),
        "Std": normal_df[col].std()
    })

sensor_range_df = pd.DataFrame(sensor_ranges)
sensor_range_df.to_csv(os.path.join(OUTPUT_DIR, "sensor_ranges.csv"), index=False)

print(sensor_range_df)

# ==============================
# CORRELATION ANALYSIS
# ==============================

print("\nComputing correlation matrix...")

corr = normal_df[sensor_cols].corr()

corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))

print("Correlation matrix saved.")

# ==============================
# VISUALIZATION
# ==============================

print("\nGenerating plots...")

important_sensors = ["LIT101", "LIT301", "LIT401", "FIT601"]

for sensor in important_sensors:
    if sensor in normal_df.columns:

        plt.figure(figsize=(12,5))

        plt.plot(normal_df[sensor][:2000], label="Normal")
        plt.plot(attack_df[sensor][:2000], label="Attack")

        plt.title(f"{sensor} Normal vs Attack")
        plt.legend()

        plt.savefig(os.path.join(OUTPUT_DIR, f"{sensor}_comparison.png"))

        plt.close()

print("Plots saved.")

# ==============================
# SAVE CLEAN FEATURE LIST
# ==============================

important_features = [
    "LIT101", "MV101", "P101",
    "LIT301", "MV301", "P301",
    "LIT401", "P401",
    "PIT501", "P501",
    "FIT601", "P601"
]

pd.Series(important_features).to_csv(
    os.path.join(OUTPUT_DIR, "recommended_features.csv"),
    index=False
)

print("\nRecommended features saved.")

# ==============================
# COMPLETION
# ==============================

print("\nEDA COMPLETE.")
print(f"All outputs saved in: {OUTPUT_DIR}")
print("="*60)
