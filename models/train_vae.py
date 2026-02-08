import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

print("="*50)
print("GenTwin VAE Training Started")
print("="*50)

# ======================
# CONFIG
# ======================

DATA_PATH = r"C:\Users\girik\Desktop\IITK_CR2\datasets\clean_swat.csv"
MODEL_SAVE_PATH = r"C:\Users\girik\Desktop\IITK_CR2\models\vae_model.pth"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
LATENT_DIM = 4

# ======================
# LOAD DATA
# ======================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

data = df.values.astype(np.float32)

input_dim = data.shape[1]

print(f"Dataset shape: {data.shape}")
print(f"Input features: {input_dim}")

tensor_data = torch.tensor(data)

dataset = TensorDataset(tensor_data)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# VAE MODEL
# ======================

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)

        # Decoder
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

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


# ======================
# LOSS FUNCTION
# ======================

def loss_function(recon_x, x, mu, logvar):

    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


# ======================
# INITIALIZE MODEL
# ======================

print("\nInitializing model...")

model = VAE(input_dim, LATENT_DIM)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======================
# TRAINING LOOP
# ======================

print("\nTraining started...\n")

for epoch in range(EPOCHS):

    total_loss = 0

    for batch in loader:

        x = batch[0]

        optimizer.zero_grad()

        reconstructed, mu, logvar = model(x)

        loss = loss_function(reconstructed, x, mu, logvar)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataset)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")


# ======================
# SAVE MODEL
# ======================

torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\nModel saved at:")
print(MODEL_SAVE_PATH)

print("\nTraining Complete.")
print("="*50)
