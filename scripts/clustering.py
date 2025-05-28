##### Base libs
import numpy as np
import pandas as pd
from tqdm import tqdm 
from random import seed, randint
from collections import Counter 
import matplotlib.pyplot as plt
import seaborn as sns

##### Dimensionality reduction for viz
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

##### Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

##### Clustering
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

##### Seed to ensure reproducibility
SEED = 42

np.random.seed(SEED)
seed(SEED)



# ------------------- CLUSTERING functions -------------------
    
    

def optimal_k(values, k_range, l2_norm = False, fix_k = None, n_init = 5):
    # store SSE for each k
    sse = []

    if l2_norm:
        # normalize to unit norm (closer to cosine similarity)
        values = normalize(values, "l2")

    for k in k_range:
        kmeans = KMeans(n_clusters = k, random_state = SEED, n_init = n_init)
        kmeans.fit(values)
        sse.append(kmeans.inertia_)

    # elbow point
    knee_locator = KneeLocator(k_range, sse, curve = "convex", direction = "decreasing")
    opt_k = knee_locator.elbow

    if fix_k:
        opt_k = fix_k

    # plot
    sns.set(style = "whitegrid", context = "talk")
    plt.figure(figsize = (10, 5))
    plt.plot(
        k_range,
        sse,
        marker = "o",
        linestyle = "--",
        color = "darkblue",
        linewidth = 2,
        markersize = 6
    )

    # vertical line for optimal k
    plt.axvline(opt_k, color = "darkred", linestyle = "-", linewidth = 2.5, label = f"Optimal k")

    xtick_labels = list(range(min(k_range) + 1, max(k_range) + 2, 2))
    plt.xticks(ticks = xtick_labels, fontsize = 18)

    # axis labels and styling
    plt.xlabel("Number of Clusters (k)", fontsize = 20)
    plt.ylabel("Sum of Squared Errors (SSE)", fontsize = 20)
    plt.yticks(fontsize = 18)
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.legend(fontsize = 18)
    plt.tight_layout()
    plt.show()

    return opt_k