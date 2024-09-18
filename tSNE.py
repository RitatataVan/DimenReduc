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
cleaned_df = cleaned_df.to_numpy()

tsne_train_loss_history = []
tsne_test_loss_history = []

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
        tsne_train_loss_history.append(train_mse)

        # Evaluating on the validation set
        model.eval()
        test_mse = 0.0
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_dataloader:
                y_pred = model.forward(X_test_batch)
                batch_mse = MSE(y_pred, y_test_batch).item()
                test_mse += batch_mse * len(X_test_batch)

        test_mse /= len(X_test)
        tsne_test_loss_history.append(test_mse)

        if train_loss_ma is None:
            train_loss_ma = train_mse
        else:
            train_loss_ma = (train_loss_ma * epoch + train_mse) / (epoch + 1)

        if test_loss_ma is None:
            test_loss_ma = test_mse
        else:
            test_loss_ma = (test_loss_ma * epoch + test_mse) / (epoch + 1)

        if epoch > 0:
            if len(tsne_test_loss_history) >= 2 and \
                    abs(test_loss_ma - tsne_test_loss_history[-2]) < convergence_threshold:
                consecutive_epochs_no_improvement += 1
                if consecutive_epochs_no_improvement >= 5:
                    print(f"Model converges at {epoch}th epoch")
                    break
            else:
                consecutive_epochs_no_improvement = 0

        if test_mse < best_mse:
            best_mse = test_mse
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "/home/fanqiany/data/fanqiany/tSNE.pth")
            print(f"Epoch {epoch + 1}/{num_epoch}, Test MSE: {test_mse:.6f}")

    return best_mse


def tsne_model(trial):
    torch.manual_seed(42)
    # perplexity = trial.suggest_int("perplexity", 60, 100, step=5)
    # learning_rate = trial.suggest_float("learning_rate", 250, 600, step=50)
    # perplexity = trial.suggest_int("perplexity", 70, 70)
    # learning_rate = trial.suggest_float("learning_rate", 400, 400)

    tsne = TSNE(n_components=2, perplexity=70, learning_rate=400)
    tsne_results = tsne.fit_transform(cleaned_df)
    np.save("/home/fanqiany/data/fanqiany/tsne_results.npy", tsne_results)

    return tsne_results


# Create study objects and optimize for t-SNE
tsne_study = optuna.create_study(direction='minimize')
with tqdm(total=1, desc="Optimizing t-SNE") as pbar:
    def callback(study, trial):
        pbar.update(1)
    tsne_study.optimize(lambda trial: objective(trial, tsne_model(trial), cleaned_df), n_trials=1, callbacks=[callback])
pbar.close()

# Print best parameters and results for t-SNE
print("Best t-SNE Parameters:", tsne_study.best_params)
print("Best t-SNE MSE:", tsne_study.best_value)

# Explained variance
cov_matrix = np.cov(cleaned_df.T)
explained_var = 1 - 19 * tsne_study.best_value / np.trace(cov_matrix)
print("Explained Variance of t-SNE:", explained_var)

print("Train History:", tsne_train_loss_history)
print("Test History:", tsne_test_loss_history)


# Plot t-SNE
def plot_tsne(data):
    tsne = TSNE(n_components=2, perplexity=70, learning_rate=400, n_iter=1000)
    tsne_results = tsne.fit_transform(data)
    kl_divergence = tsne.kl_divergence_

    plt.figure(figsize=(10, 8))
    hist = plt.hist2d(tsne_results[:, 0], tsne_results[:, 1], bins=(50, 50), cmap='viridis')
    plt.colorbar(hist[3], label='Frequency')
    plt.title(f't-SNE of Chemical Abundances (perplexity: {70}, KL Div: {kl_divergence:.2f})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig("/home/fanqiany/data/fanqiany/tSNE.png", dpi=300, bbox_inches='tight')
    plt.close()


plot_tsne(cleaned_df)

