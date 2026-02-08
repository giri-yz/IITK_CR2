import torch
import torch.nn as nn
import pandas as pd
import numpy as np

print("="*50)
print("Generating Strong Synthetic Attacks")
print("="*50)

MODEL_PATH = r"C:\Users\girik\Desktop\IITK_CR2\models\vae_model.pth"
DATA_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\clean_swat.csv"
OUTPUT_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\synthetic_attack.csv"

LATENT_DIM = 4
NUM_SAMPLES = 5000


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

    def decode(self, z):
        return self.decoder(z)


# Load model
df = pd.read_csv(DATA_PATH)
input_dim = df.shape[1]

model = VAE(input_dim, LATENT_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# Generate base normal samples
z = torch.randn(NUM_SAMPLES, LATENT_DIM)
synthetic = model.decode(z).detach().numpy()


# STRONG ATTACK INJECTION
attack = synthetic.copy()

# Tank level attack (extreme low/high)
attack[:, 1] = np.random.choice([0.0, 1.0], NUM_SAMPLES)

# Flow sensor attack
attack[:, 0] = np.random.choice([0.0, 1.0], NUM_SAMPLES)

# Pressure attack
attack[:, 6] = np.random.choice([0.0, 1.0], NUM_SAMPLES)

# Pump malfunction attack
attack[:, 8] = np.random.choice([0, 1], NUM_SAMPLES)

# Random extreme corruption
mask = np.random.rand(*attack.shape) < 0.3
attack[mask] = np.random.choice([0.0, 1.0], size=np.sum(mask))

attack_df = pd.DataFrame(attack, columns=df.columns)
attack_df.to_csv(OUTPUT_PATH, index=False)

print("Strong synthetic attacks generated.")
