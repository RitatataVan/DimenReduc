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
# 10% of total dataset
# rows_to_select = int(0.1 * len(cleaned_df))
# cleaned_df = cleaned_df.sample(n=rows_to_select, random_state=42)
cleaned_df = cleaned_df.to_numpy()

umap_train_loss_history = []
umap_test_loss_history = []


def objective(trial, X, y):
    print("Executing nn_objective...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    torch.manual_seed(42)
    model = NNModel(2, 64, 32, 19)
    MSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_mse = float('inf')
    best_weights = None

    train_loss_ma = None
    test_loss_ma = None
    convergence_threshold = 0.0001
    consecutive_epochs_no_improvement = 0

    # Training
    num_epoch = 3000
    for epoch in range(num_epoch):
        model.train()
        train_mse = 0.0
        for X_train_batch, y_train_batch in train_dataloader:
            optimizer.zero_grad()
            predictions = model.forward(X_train_batch)
            loss_value = MSE(predictions, y_train_batch)
            loss_value.backward()
            optimizer.step()
            train_mse += loss_value.item() * len(X_train_batch)

        train_mse /= len(X_train)
        umap_train_loss_history.append(train_mse)

        # Evaluating on the validation set
        model.eval()
        test_mse = 0.0
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_dataloader:
                y_pred = model.forward(X_test_batch)
                batch_mse = MSE(y_pred, y_test_batch).item()
                test_mse += batch_mse * len(X_test_batch)

        test_mse /= len(X_test)
        umap_test_loss_history.append(test_mse)

        if train_loss_ma is None:
            train_loss_ma = train_mse
        else:
            train_loss_ma = (train_loss_ma * epoch + train_mse) / (epoch + 1)

        if test_loss_ma is None:
            test_loss_ma = test_mse
        else:
            test_loss_ma = (test_loss_ma * epoch + test_mse) / (epoch + 1)

        if epoch > 0:
            if abs(test_loss_ma - umap_test_loss_history[-2]) < convergence_threshold:
                consecutive_epochs_no_improvement += 1
                if consecutive_epochs_no_improvement >= 5:
                    print(f"Model converges at {epoch}th epoch")
                    break
            else:
                consecutive_epochs_no_improvement = 0

        if test_mse < best_mse:
            best_mse = test_mse
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "/home/fanqiany/data/fanqiany/UMAP.pth")
            print(f"Epoch {epoch + 1}/{num_epoch}, Test MSE: {test_mse:.6f}")

    return best_mse


def umap_model(trial):
    torch.manual_seed(42)
    # n_neighbors = trial.suggest_int('n_neighbors', 30, 90, step=10)
    # min_dist = trial.suggest_float('min_dist', 0.001, 0.025, step=0.004)
    # n_neighbors = trial.suggest_int('n_neighbors', 80, 80)
    # min_dist = trial.suggest_float('min_dist', 0.005, 0.005)

    Umap_model = umap.UMAP(n_neighbors=80, min_dist=0.005, n_components=2, random_state=42)
    umap_results = Umap_model.fit_transform(cleaned_df)
    np.save("/home/fanqiany/data/fanqiany/UMAP_results.npy", umap_results)

    return umap_results


# Create study objects and optimize for UMAP
umap_study = optuna.create_study(direction='minimize')
with tqdm(total=1, desc="Optimizing UMAP") as pbar:
    def callback(study, trial):
        pbar.update(1)
    umap_study.optimize(lambda trial: objective(trial, umap_model(trial), cleaned_df), n_trials=1, callbacks=[callback])
pbar.close()


# Print best parameters and results for UMAP
print("Best UMAP Parameters:", umap_study.best_params)
print("Best UMAP MSE:", umap_study.best_value)

# Explained variance
cov_matrix = np.cov(cleaned_df.T)
explained_var = 1 - 19 * umap_study.best_value / np.trace(cov_matrix)
print("Explained Variance of UMAP:", explained_var)

print("Train History:", umap_train_loss_history)
print("Test History:", umap_test_loss_history)


# Plot UMAP
def plot_umap(data):
    Umap_model = umap.UMAP(n_neighbors=80, min_dist=0.005, n_components=2, random_state=42)
    umap_results = Umap_model.fit_transform(data)

    plt.figure(figsize=(10, 8))
    hist = plt.hist2d(umap_results[:, 0], umap_results[:, 1], bins=(50, 50), cmap='viridis')
    plt.colorbar(hist[3], label='Frequency')
    plt.title(f'UMAP of Chemical Abundances (n_neighbors: {80}, min_dist: {0.005})')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.savefig("/home/fanqiany/data/fanqiany/UMAP.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_umap(cleaned_df)
