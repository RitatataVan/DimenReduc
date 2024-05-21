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


# original data for plot (color by mean FE_H)
df = pd.read_csv("/home/fanqiany/data/fanqiany/APOGEEDR17_GAIAEDR3_noflagfilter.csv")

# prepare variables
chemical_abundances = ['FE_H', 'C_FE', 'CI_FE', 'N_FE', 'O_FE', 'MG_FE', 'AL_FE',
                       'SI_FE', 'P_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'TIII_FE',
                       'V_FE', 'CR_FE', 'MN_FE', 'CO_FE', 'NI_FE']

# remove 'ASPCAPFLAG' and 'STARFLAG'
selected_df = df[(df['ASPCAPFLAG'] == 0) & (df['STARFLAG'] == 0)][chemical_abundances]

# remove outliers if exist
z_scores = abs((selected_df - selected_df.mean()) / selected_df.std())
data_no_outliers = selected_df[(z_scores < 3).all(axis=1)]

# save cleaned dataset
df_color_mean_feh = pd.DataFrame(data_no_outliers, columns=chemical_abundances)
df_color_mean_feh = df_color_mean_feh.to_numpy()


# Plot PCA by mean of FE_H
def plot_scatter_pca(data):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Calculate mean FE_H for each point
    fe_h_values = data[:, 0]  # first column is FE_H

    # mean_fe_h = []
    # for i in range(len(tsne_results)):
    #     indices = np.isclose(tsne_results[:, :2], tsne_results[i, :2]).all(axis=1)
    #     mean_fe_h.append(fe_h_values[indices].mean().item())
    #
    # mean_fe_h = np.array(mean_fe_h)

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=fe_h_values, cmap='viridis', marker='o', alpha=0.5, s=1)
    plt.colorbar(label='Mean FE_H')
    plt.title(f'PCA of Chemical Abundances (perplexity: {70})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.savefig("/home/fanqiany/data/fanqiany/PCA_FE_H.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_scatter_pca(df_color_mean_feh)

# Plot t-SNE by mean of FE_H
def plot_scatter_tsne(data):
    tsne_results = np.load("/home/fanqiany/data/fanqiany/tsne_results.npy")

    # Calculate mean FE_H for each point
    fe_h_values = data[:, 0]  # first column is FE_H

    # mean_fe_h = []
    # for i in range(len(tsne_results)):
    #     indices = np.isclose(tsne_results[:, :2], tsne_results[i, :2]).all(axis=1)
    #     mean_fe_h.append(fe_h_values[indices].mean().item())
    #
    # mean_fe_h = np.array(mean_fe_h)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=fe_h_values, cmap='viridis', marker='o', alpha=0.5, s=1)
    plt.colorbar(label='Mean FE_H')
    plt.title(f't-SNE of Chemical Abundances (perplexity: {70})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig("/home/fanqiany/data/fanqiany/tSNE_FE_H.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_scatter_tsne(df_color_mean_feh)


# Plot UMAP by mean of FE_H
def plot_scatter_umap(data):
    umap_results = np.load("/home/fanqiany/data/fanqiany/UMAP_results.npy")

    # Calculate mean FE_H for each point
    fe_h_values = data[:, 0]  # first column is FE_H

    # mean_fe_h = []
    # for i in range(len(umap_results)):
    #     indices = np.isclose(umap_results[:, :2], umap_results[i, :2]).all(axis=1)
    #     mean_fe_h.append(fe_h_values[indices].mean().item())
    #
    # mean_fe_h = np.array(mean_fe_h)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=fe_h_values, cmap='viridis', marker='o', alpha=0.5, s=1)
    plt.colorbar(label='Mean FE_H')
    plt.title(f'UMAP of Chemical Abundances (n_neighbors: {80}, min_dist: {0.005})')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.savefig("/home/fanqiany/data/fanqiany/UMAP_FE_H.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_scatter_umap(df_color_mean_feh)


# Plot AE by mean of FE_H
def plot_scatter_ae(data):
    data = torch.tensor(data, dtype=torch.float32)
    model = AutoEncoder(input_size=19, latent_size=2)
    model.load_state_dict(torch.load("/home/fanqiany/data/fanqiany/AE.pth"))

    model.eval()
    with torch.no_grad():
        latent = model.encoder(data)

    # Calculate mean FE_H for each point
    fe_h_values = data[:, 0]  # first column is FE_H

    mean_fe_h = []
    for i in range(len(latent)):
        distance = np.linalg.norm(latent - latent[i], axis=1)
        mean_fe_h.append(np.mean(fe_h_values[distance < 0.1].numpy(), dtype=np.float32))

    mean_fe_h = np.array(mean_fe_h)

    plt.figure(figsize=(10, 8))
    plt.scatter(latent[:, 0], latent[:, 1], c=mean_fe_h, cmap='viridis', marker='o', alpha=0.5, s=1)
    plt.colorbar(label='Mean FE_H')
    plt.title('AutoEncoder of Chemical Abundances')
    plt.xlabel('Latent Component 1')
    plt.ylabel('Latent Component 2')
    plt.grid(True)
    plt.xlim(latent[:, 0].min(), latent[:, 0].max())
    plt.ylim(latent[:, 1].min(), latent[:, 1].max())
    plt.savefig("/home/fanqiany/data/fanqiany/AE_FE_H.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_scatter_ae(df_color_mean_feh)


# Plot VAE by mean of FE_H
def plot_scatter_vae(data):
    data = torch.tensor(data, dtype=torch.float32)
    model = VAE(input_size=19, latent_size=2)
    model.load_state_dict(torch.load("/home/fanqiany/data/fanqiany/VAE.pth"))

    model.eval()
    with torch.no_grad():
        x = model.encoder(data)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        latent = model.reparameterize(mu, logvar)

    # Calculate mean FE_H for each point
    fe_h_values = data[:, 0]  # first column is FE_H

    mean_fe_h = []
    for i in range(len(latent)):
        distance = np.linalg.norm(latent - latent[i], axis=1)
        mean_fe_h.append(np.mean(fe_h_values[distance < 0.1].numpy(), dtype=np.float32))

    mean_fe_h = np.array(mean_fe_h)

    plt.figure(figsize=(10, 8))
    plt.scatter(latent[:, 0], latent[:, 1], c=mean_fe_h, cmap='viridis', marker='o', alpha=0.5, s=1)
    plt.colorbar(label='Mean FE_H')
    plt.title('VAE of Chemical Abundances')
    plt.xlabel('Latent Component 1')
    plt.ylabel('Latent Component 2')
    plt.grid(True)
    plt.xlim(latent[:, 0].min(), latent[:, 0].max())
    plt.ylim(latent[:, 1].min(), latent[:, 1].max())
    plt.savefig("/home/fanqiany/data/fanqiany/VAE_FE_H.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_scatter_vae(df_color_mean_feh)
