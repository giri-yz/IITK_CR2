import torch
import torch.nn as nn
import pandas as pd
import numpy as np

print("="*50)
print("Testing VAE Model")
print("="*50)

MODEL_PATH = r"C:\Users\girik\Desktop\IITK_CR2\models\vae_model.pth"
NORMAL_DATA = r"C:\Users\girik\Desktop\IITK_CR2\datasets\clean_swat.csv"
ATTACK_DATA = r"C:\Users\girik\Desktop\IITK_CR2\datasets\synthetic_attack.csv"

LATENT_DIM = 4


# VAE Definition
class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

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
        reconstructed = self.decoder(z)
        return reconstructed


# Load normal data
normal_df = pd.read_csv(NORMAL_DATA)
normal_tensor = torch.tensor(normal_df.values.astype(np.float32))

# Load attack data
attack_df = pd.read_csv(ATTACK_DATA)
attack_tensor = torch.tensor(attack_df.values.astype(np.float32))

input_dim = normal_tensor.shape[1]

# Load model
model = VAE(input_dim, LATENT_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("\nModel loaded successfully.")

# Test normal data
with torch.no_grad():

    normal_recon = model(normal_tensor)
    normal_error = torch.mean((normal_tensor - normal_recon)**2).item()

    attack_recon = model(attack_tensor)
    attack_error = torch.mean((attack_tensor - attack_recon)**2).item()

print("\nNormal reconstruction error:", normal_error)
print("Attack reconstruction error:", attack_error)

if attack_error > normal_error:
    print("\nModel is WORKING correctly.")
else:
    print("\nModel may need improvement.")

print("="*50)
