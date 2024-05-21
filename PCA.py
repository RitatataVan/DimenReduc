import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def pca(data):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    np.save("/home/fanqiany/data/fanqiany/pca_result.npy", pca_result)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC 1', 'PC 2'])
    explained_variance_ratio = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    hist = plt.hist2d(pca_df['PC 1'], pca_df['PC 2'], bins=(50, 50), cmap='viridis')
    plt.colorbar(hist[3], label='Frequency')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(f'PCA of Chemical Abundances\nExplained Variance Ratio: PC1 = {explained_variance_ratio[0]:.2%}, PC2 = {explained_variance_ratio[1]:.2%}')
    plt.grid(True)
    plt.savefig("/home/fanqiany/data/fanqiany/PCA.png", dpi=300, bbox_inches='tight')
    plt.close()


pca(cleaned_df)
