import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv(r"C:\Users\girik\Desktop\IITK_CR2\datasets\normal.csv")

# Select safe features
features = [
    "FIT101",
    "LIT101",
    "FIT301",
    "LIT301",
    "FIT401",
    "LIT401",
    "PIT501",
    "FIT601",
    "P101",
    "MV301",
    "P301",
    "P401",
    "P501",
    "P601"
]

df = df[features]

# Take subset for fast training
df = df.iloc[:50000]

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Save clean dataset
clean_df = pd.DataFrame(scaled_data, columns=features)
clean_df.to_csv("clean_swat.csv", index=False)

print("Clean dataset saved.")
print(clean_df.shape)
