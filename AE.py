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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm


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
cleaned_df = cleaned_df.to_numpy()

ae_train_loss_history = []
ae_test_loss_history = []

def autoencoder_objective(X):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, X_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, X_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    torch.manual_seed(42)
    model = AutoEncoder()
    MSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_mse = float('inf')
    best_weights = None

    # train_loss_ma = None
    # test_loss_ma = None
    # convergence_threshold = 0.0001
    # consecutive_epochs_no_improvement = 0

    # Training
    num_epoch = 5000
    for epoch in range(num_epoch):
        model.train()
        train_mse = 0.0
        for X_train_batch in train_dataloader:
            optimizer.zero_grad()
            # Extract input from tuple
            X_train_batch = X_train_batch[0]
            # Get encoder output
            latent_space = model.encoder(X_train_batch)
            # Use latent space for reconstruction in the decoder
            predictions = model.decoder(latent_space)
            loss_value = MSE(predictions, X_train_batch)
            loss_value.backward()
            optimizer.step()
            train_mse += loss_value.item() * len(X_train_batch)

        train_mse /= len(X_train)
        ae_train_loss_history.append(train_mse)

        # Evaluating on the validation set
        model.eval()
        test_mse = 0.0
        with torch.no_grad():
            for X_test_batch in test_dataloader:
                # Extract input from tuple
                X_test_batch = X_test_batch[0]
                # Get encoder output for the test set
                latent_space = model.encoder(X_test_batch)
                # Use latent space for reconstruction in the decoder
                pred = model.decoder(latent_space)
                batch_mse = MSE(pred, X_test_batch).item()
                test_mse += batch_mse * len(X_test_batch)

        test_mse /= len(X_test)
        ae_test_loss_history.append(test_mse)

        # if train_loss_ma is None:
        #     train_loss_ma = train_mse
        # else:
        #     train_loss_ma = (train_loss_ma * epoch + train_mse) / (epoch + 1)
        #
        # if test_loss_ma is None:
        #     test_loss_ma = test_mse
        # else:
        #     test_loss_ma = (test_loss_ma * epoch + test_mse) / (epoch + 1)
        #
        # if epoch > 0:
        #     if abs(test_loss_ma - ae_test_loss_history[-2]) < convergence_threshold:
        #         consecutive_epochs_no_improvement += 1
        #         if consecutive_epochs_no_improvement >= 5:
        #             print(f"Model converges at {epoch}th epoch")
        #             break
        #     else:
        #         consecutive_epochs_no_improvement = 0

        if test_mse < best_mse:
            best_mse = test_mse
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "/home/fanqiany/data/fanqiany/AE1.pth")
            # torch.save(model.state_dict(), "/home/fanqiany/data/fanqiany/AE.pth")
            print(f"Epoch {epoch + 1}/{num_epoch}, Test MSE: {test_mse:.6f}")

    return best_mse


best_mse = autoencoder_objective(cleaned_df)
print("Best AutoEncoder MSE:", best_mse)

# Explained variance
cov_matrix = np.cov(cleaned_df.T)
explained_var = 1 - 19 * best_mse / np.trace(cov_matrix)
print("Explained Variance of AutoEncoder:", explained_var)

print("Train History:", ae_train_loss_history)
print("Test History:", ae_test_loss_history)


# Plot AE
def plot_ae(data):
    data = torch.tensor(data, dtype=torch.float32)
    model = AutoEncoder(input_size=19, latent_size=2)

    model.eval()
    with torch.no_grad():
        latent = model.encoder(data)

    plt.figure(figsize=(10, 8))
    plt.hist2d(latent[:, 0], latent[:, 1], bins=(50, 50), cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.title('AutoEncoder of Chemical Abundances')
    plt.xlabel('Latent Component 1')
    plt.ylabel('Latent Component 2')
    plt.grid(True)
    plt.xlim(latent[:, 0].min(), latent[:, 0].max())
    plt.ylim(latent[:, 1].min(), latent[:, 1].max())
    plt.savefig("/home/fanqiany/data/fanqiany/AE.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_ae(cleaned_df)
