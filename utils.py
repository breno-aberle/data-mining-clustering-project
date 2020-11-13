import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def zscore(x, mean, std):
    score = (x - mean) / std
    return score

def min_max_scaling():
    return

def mean_normalization():
    return 


# Returns dict
def metrics(data, idx_cols=0):
    # get parameters
    min_max_values = [(np.min(data[col].values), np.max(data[col].values), np.mean(data[col].values), \
                       np.std(data[col].values)) for col in data.columns[idx_cols:]]
    
    # create dictionary
    keys = data.columns[idx_cols:]
    dict_attributes = dict(zip(keys, (min_max_values)))

    # create with dataframe
    df_metrics = pd.DataFrame(data=dict_attributes).T
    df_metrics.columns = ['min', 'max', 'mean', 'std']

    return df_metrics

# Returns df
def standardize(data, df_metrics, idx_cols=0):
    # Apply standardization on genedata dataset
    # zscore

    data_std = pd.DataFrame()

    for col in data.columns[idx_cols:]:
        values_std = zscore(data[col].values, df_metrics.loc[col]['mean'], df_metrics.loc[col]['std'])
        data_std[col] = values_std
    
    return data_std

def dim_reduce(data, pct=0.99):
    pca = PCA(n_components=pct)
    pca.fit(data)
    return pd.DataFrame(pca.transform(data))

# Returns df
def preprocess(data, idx_cols=0, std_pca=False):
    
    metrics_dict = metrics(data, idx_cols)
    data_std = standardize(data, metrics_dict, idx_cols)
    data_reduced = dim_reduce(data_std)
    pca_metrics_dict = metrics(data_reduced)
    pca_std = standardize(data_reduced, pca_metrics_dict)
    if std_pca:
        return pca_std
    else:
        return data_reduced


def plot_hist(data, bins=30, cols=3):
    keys = data.columns.values
    for i in range(cols):
        plt.hist(data[keys[i]], bins=bins)
        plt.xlabel('Z-score')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of data feature {i}')
        plt.show()

def kmeans(data, k):
    km = KMeans(n_clusters=k, n_init=15, random_state=123, tol=1e-6)
    preds = km.fit_predict(data)
    return preds, km

def nmi(truth, preds, c, d=5):
    # preds = c
    # truth = d
    n = len(truth)
    
    pcd = np.zeros((c, d)) # cluster x classification
    pc = np.zeros(c)
    pd = np.zeros(d)
    
    hc = 0
    hd = 0

    for i in range(len(truth)):
        pcd[preds[i]][truth[i]] += 1 / n
        pc[preds[i]] += 1 / n
        pd[truth[i]] += 1 / n


    mi = 0
    for i in range(c):
        for j in range(d):
            if pcd[i][j] / (pc[i] * pd[j]) != 0:
                mi += pcd[i][j] * np.log(pcd[i][j] / (pc[i] * pd[j]))
    
    
    for i in range(c):
        hc += pc[i] * np.log(pc[i])

    for i in range(d):
        hd += pd[i] * np.log(pd[i])

    den = np.sqrt(hc * hd)

    return mi / den

    
