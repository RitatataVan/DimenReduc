import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import os
import copy
import umap
import seaborn as sns
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm


# NN
class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NNModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x


# AE
class AutoEncoder(nn.Module):
    def __init__(self, input_size=19, latent_size=2):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_size)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


# VAE
class VAE(nn.Module):
    def __init__(self, input_size=19, latent_size=2, kld_weight=0.00025):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.kld_weight = kld_weight
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_var = nn.Linear(64, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, input_size)
        )

    def encode(self, x):
        x = self.encoder(x)
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Split the encoder result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # backpropagate
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def vae_loss(self, x_recon, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # KL divergence
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return recon_loss + kl_divergence * self.kld_weight


# Data
df = pd.read_csv("/home/fanqiany/data/fanqiany/APOGEEDR17_GAIAEDR3_noflagfilter.csv")
chemical_abundances = ['FE_H', 'C_FE', 'CI_FE', 'N_FE', 'O_FE', 'MG_FE', 'AL_FE',
                       'SI_FE', 'P_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'TIII_FE',
                       'V_FE', 'CR_FE', 'MN_FE', 'CO_FE', 'NI_FE']

# remove 'ASPCAPFLAG' and 'STARFLAG'
selected_df = df[(df['ASPCAPFLAG'] == 0) & (df['STARFLAG'] == 0)][chemical_abundances]

# remove outliers if exist
z_scores = abs((selected_df - selected_df.mean()) / selected_df.std())
data_no_outliers = selected_df[(z_scores < 3).all(axis=1)]

# standard scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_no_outliers)

# save cleaned dataset
cleaned_df = pd.DataFrame(scaled_data, columns=chemical_abundances)

# selected row
selected_row = cleaned_df.iloc[1000]
selected_df_np = torch.tensor(selected_row.values, dtype=torch.float32)


# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cleaned_df)
selected_pca = pca_result[1000]

recon_data = pca.inverse_transform(selected_pca.reshape(1, -1))

selected_df_np = selected_df_np.squeeze().numpy()
recon_data_np = recon_data.squeeze()

plt.figure(figsize=(10, 6))
plt.plot(chemical_abundances, selected_df_np, label='Original Data', marker='o')
plt.plot(chemical_abundances, recon_data_np, label='PCA Reconstructed Data', marker='x')
plt.title('Comparison of Original and PCA Reconstructed Data')
plt.xticks(rotation=45)
plt.ylabel('Abundance Values')
plt.legend()
plt.savefig("/home/fanqiany/data/fanqiany/TestPCA_Perfor.png", dpi=300, bbox_inches='tight')
plt.close()

# tSNE
tsne_results = np.load("/home/fanqiany/data/fanqiany/tsne_results.npy")
selected_tsne = tsne_results[1000]

model_tsne = NNModel(2, 64, 32, 19)
model_tsne.load_state_dict(torch.load("/home/fanqiany/data/fanqiany/tSNE.pth"))

model_tsne.eval()
with torch.no_grad():
    recon_data = model_tsne(torch.from_numpy(selected_tsne).unsqueeze(0))

selected_df_np = selected_df_np.squeeze()
recon_data_np = recon_data.squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.plot(chemical_abundances, selected_df_np, label='Original Data', marker='o')
plt.plot(chemical_abundances, recon_data_np, label='tSNE Reconstructed Data', marker='x')
plt.title('Comparison of Original and tSNE Reconstructed Data')
plt.xticks(rotation=45)
plt.ylabel('Abundance Values')
plt.legend()
plt.savefig("/home/fanqiany/data/fanqiany/TesttSNE_Perfor.png", dpi=300, bbox_inches='tight')
plt.close()

# UMAP
umap_results = np.load("/home/fanqiany/data/fanqiany/UMAP_results.npy")
selected_umap = umap_results[1000]

model_umap = NNModel(2, 64, 32, 19)
model_umap.load_state_dict(torch.load("/home/fanqiany/data/fanqiany/UMAP.pth"))

model_umap.eval()
with torch.no_grad():
    recon_data = model_umap(torch.from_numpy(selected_umap).unsqueeze(0))

selected_df_np = selected_df_np.squeeze()
recon_data_np = recon_data.squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.plot(chemical_abundances, selected_df_np, label='Original Data', marker='o')
plt.plot(chemical_abundances, recon_data_np, label='UMAP Reconstructed Data', marker='x')
plt.title('Comparison of Original and UMAP Reconstructed Data')
plt.xticks(rotation=45)
plt.ylabel('Abundance Values')
plt.legend()
plt.savefig("/home/fanqiany/data/fanqiany/TestUMAP_Perfor.png", dpi=300, bbox_inches='tight')
plt.close()

# AE
model = AutoEncoder(input_size=19, latent_size=2)
model.load_state_dict(torch.load("/home/fanqiany/data/fanqiany/AE.pth"))

model.eval()
with torch.no_grad():
    x_recon = model(selected_df_np.unsqueeze(0))

selected_df_np = selected_df_np.squeeze()
x_recon_np = x_recon.squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.plot(chemical_abundances, selected_df_np, label='Original Data', marker='o')
plt.plot(chemical_abundances, x_recon_np, label='AE Reconstructed Data', marker='x')
plt.title('Comparison of Original and AE Reconstructed Data')
plt.xticks(rotation=45)
plt.ylabel('Abundance Values')
plt.legend()
plt.savefig("/home/fanqiany/data/fanqiany/TestAE_Perfor.png", dpi=300, bbox_inches='tight')
plt.close()

# VAE
model = VAE(input_size=19, latent_size=2)
model.load_state_dict(torch.load("/home/fanqiany/data/fanqiany/VAE.pth"))

model.eval()
with torch.no_grad():
    x_recon, mu, logvar = model(selected_df_np.unsqueeze(0))

selected_df_np = selected_df_np.squeeze()
x_recon_np = x_recon.squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.plot(chemical_abundances, selected_df_np, label='Original Data', marker='o')
plt.plot(chemical_abundances, x_recon_np, label='VAE Reconstructed Data', marker='x')
plt.title('Comparison of Original and VAE Reconstructed Data')
plt.xticks(rotation=45)
plt.ylabel('Abundance Values')
plt.legend()
plt.savefig("/home/fanqiany/data/fanqiany/TestVAE_Perfor.png", dpi=300, bbox_inches='tight')
plt.close()
