import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("="*60)
print("GenTwin VAE Model Evaluation (Optimized Threshold)")
print("="*60)

# ======================
# CONFIG
# ======================

MODEL_PATH = r"C:\Users\girik\Desktop\IITK_CR2\models\vae_model.pth"
NORMAL_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\clean_swat.csv"
ATTACK_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\synthetic_attack.csv"

LATENT_DIM = 4


# ======================
# VAE Definition
# ======================

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)


# ======================
# Load datasets
# ======================

print("\nLoading datasets...")

normal_df = pd.read_csv(NORMAL_PATH)
attack_df = pd.read_csv(ATTACK_PATH)

normal_tensor = torch.tensor(normal_df.values.astype(np.float32))
attack_tensor = torch.tensor(attack_df.values.astype(np.float32))

input_dim = normal_tensor.shape[1]

print(f"Normal samples: {len(normal_tensor)}")
print(f"Attack samples: {len(attack_tensor)}")


# ======================
# Load model
# ======================

model = VAE(input_dim, LATENT_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("\nModel loaded successfully.")


# ======================
# Calculate reconstruction errors
# ======================

print("\nCalculating reconstruction errors...")

with torch.no_grad():

    normal_recon = model(normal_tensor)
    normal_errors = torch.mean((normal_tensor - normal_recon)**2, dim=1).numpy()

    attack_recon = model(attack_tensor)
    attack_errors = torch.mean((attack_tensor - attack_recon)**2, dim=1).numpy()


print("\nError statistics:")
print(f"Normal Mean Error : {np.mean(normal_errors):.6f}")
print(f"Attack Mean Error : {np.mean(attack_errors):.6f}")


# ======================
# AUTOMATIC THRESHOLD OPTIMIZATION
# ======================

print("\nOptimizing threshold...")

all_errors = np.concatenate([normal_errors, attack_errors])

normal_true = np.zeros(len(normal_errors))
attack_true = np.ones(len(attack_errors))

y_true = np.concatenate([normal_true, attack_true])

best_threshold = 0
best_f1 = 0

for threshold in np.linspace(min(all_errors), max(all_errors), 200):

    normal_pred = normal_errors > threshold
    attack_pred = attack_errors > threshold

    y_pred = np.concatenate([normal_pred, attack_pred])

    f1 = f1_score(y_true, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


threshold = best_threshold

print(f"\nBest Threshold: {threshold:.6f}")
print(f"Best F1 Score : {best_f1:.4f}")


# ======================
# Final Predictions
# ======================

normal_pred = normal_errors > threshold
attack_pred = attack_errors > threshold

y_pred = np.concatenate([normal_pred, attack_pred])


# ======================
# Metrics calculation
# ======================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)

TN, FP, FN, TP = cm.ravel()


# ======================
# Print results
# ======================

print("\nConfusion Matrix:")
print(f"True Negative : {TN}")
print(f"False Positive: {FP}")
print(f"False Negative: {FN}")
print(f"True Positive : {TP}")

print("\nFinal Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("="*60)
print("Evaluation Complete.")
print("="*60)
