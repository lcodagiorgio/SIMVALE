##### Base libs
import matplotlib.pyplot as plt
from random import seed, randint
import seaborn as sns
import pandas as pd
import numpy as np

##### Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

##### Dimensionality reduction for viz
from sklearn.manifold import TSNE  
from mpl_toolkits.mplot3d import Axes3D

##### Seed to ensure reproducibility
SEED = 42

seed
np.random.seed(SEED)



# ------------------- PLOTTING functions -------------------



def box_violin_plot(df, col_list, title, kind = "box"):
    df_long = df[col_list].melt(var_name = "Trait", value_name = "Score")
    plt.figure(figsize = (8, 4))
    if kind == "box":
        sns.boxplot(data = df_long, x = "Trait", y = "Score")
    elif kind == "violin":
        sns.violinplot(data = df_long, x = "Trait", y = "Score", inner = "quartile")

    plt.title(f"{title}")
    plt.xlabel("Trait")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def plot_kde(df, col_list, title = ""):
    # plot (overlapping) kde of each feature in the list
    plt.figure(figsize = (8, 4))
    for col in col_list:
        sns.kdeplot(df[col], label = col, fill = True, alpha = 0.4, clip = (0, 100))

    plt.title(f"{title}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()



def plot_cond_kde(df, col_list, cat_col):
    for col in col_list:
        plt.figure(figsize = (8, 4))
        for cat in df[cat_col].unique():
            subset = df[df[cat_col] == cat][col]
            sns.kdeplot(subset, label = f"{cat}", fill = True, alpha = 0.4, clip = (0, 100))
            
        plt.title(f"{col} vs {cat_col}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()



def plot_hist(df, col_list, title = ""):
    # plot histogram of each feature in the list
    for col in col_list:
        plt.figure(figsize = (8, 4))
        sns.histplot(df[col], label = col, fill = True, alpha = 0.4, bins = 50)

        plt.title(f"{title}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()



def plot_counts(df, col, title = ""):
    sns.set(style = "whitegrid", context = "talk")
    plt.figure(figsize = (10, 6))

    sns.countplot(x = df[col], palette = "viridis")

    # axis labels and styling
    plt.xlabel(col.capitalize(), fontsize = 32)
    plt.ylabel("Count", fontsize = 32)
    plt.xticks(fontsize = 28)
    plt.yticks(fontsize = 28)
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.tight_layout()
    plt.show()



def apply_tsne(values_1, labels_1, values_2 = None, labels_2 = None, n_components = 2):
    # apply t-SNE on the embeddings and convert to df for visualization
    # also connotate each embedding with labels
    if (values_2 is not None) or (labels_2 is not None):
        values = np.vstack((values_1, values_2))
        labels = np.vstack((labels_1, labels_2))
    else:
        values = values_1
        labels = labels_1
    
    tsne = TSNE(n_components = n_components, random_state = SEED)
    tsne_embs = tsne.fit_transform(values)
    
    # create dataframe
    dim_cols = [f"Dimension {i + 1}" for i in range(n_components)]
    df_tsne = pd.DataFrame(tsne_embs, columns = dim_cols)
    df_tsne["label"] = labels
    
    return df_tsne



def plot_tsne(df, col = "label", title = "", colors = None):      
    # plot
    plt.figure(figsize = (8, 6))
    
    if colors:
        palette = colors
    else:
        palette = ["forestgreen", "goldenrod", "royalblue", "orangered", "purple", "cyan"]
    
    sns.scatterplot(data = df, x = "Dimension 1", y = "Dimension 2", hue = col, 
                    alpha = 0.7, s = 60, edgecolor = "black", palette = palette)

    plt.title(f"{title}")
    plt.legend()
    plt.show()
    
    
    
def plot_tsne_3d(df, col = "label", title = "", colors = None):
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")

    labels = df[col].unique()
    
    if colors:
        palette = colors
    else:
        palette = sns.color_palette("husl", len(labels))

    for label, color in zip(labels, palette):
        subset = df[df[col] == label]
        ax.scatter(subset["Dimension 1"], subset["Dimension 2"], subset["Dimension 3"],
                   label = label, color = color, s = 60, alpha = 0.7, edgecolor = "black")

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend()
    plt.show() 
    


def plot_std_centroids(df, analysis_cols, clust_col, title = "Toxicity Profiles per Cluster"):
    # style
    sns.set(style = "whitegrid", context = "talk")

    # Define custom color mapping for features
    color_map = {
        "obscene": "#1f4e79",          # dark blue
        "threat": "#b58900",           # dark yellow / goldenrod
        "insult": "#2e7d32",           # dark green
        "identity_attack": "#a40000"   # dark red
    }

    # standardize the toxicity features
    scaler = StandardScaler()
    df_tox_norm = df.copy()
    df_tox_norm[analysis_cols] = scaler.fit_transform(df[analysis_cols])

    # standardized toxic centroids
    tox_norm_centroids = df_tox_norm.groupby(clust_col)[analysis_cols].mean()

    plt.figure(figsize = (10, 5))
    sns.heatmap(
        tox_norm_centroids,
        annot = True,
        cmap = "RdBu_r",
        center = 0,
        cbar_kws = {"label": "Mean Standardized Values"},
        linewidths = 0.5,
        linecolor = "white",
        annot_kws = {"fontsize": 12})
    
    plt.title(title, fontsize = 20)
    plt.xlabel("Type", fontsize = 18)
    plt.ylabel("Cluster", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.tight_layout()
    plt.show()

    # bar plot with transparency
    # bar Plot with conditional transparency
    fig, ax = plt.subplots(figsize = (10, 5))
    cluster_indices = tox_norm_centroids.index
    bar_width = 0.15
    positions = np.arange(len(cluster_indices))

    # keep track of labels to avoid duplicates
    plotted_labels = set()

    for i, feature in enumerate(analysis_cols):
        values = tox_norm_centroids[feature]
        color = color_map[feature]
        alphas = [0.5 if v < 0 else 1.0 for v in values]

        for j, (val, alpha) in enumerate(zip(values, alphas)):
            ax.bar(
                positions[j] + i * bar_width,
                val,
                width = bar_width,
                color = color,
                alpha = alpha,
                label = feature if feature not in plotted_labels else "")
            
        plotted_labels.add(feature)

    # axis formatting
    ax.set_xticks(positions + bar_width * (len(analysis_cols) - 1) / 2)
    ax.set_xticklabels(cluster_indices, fontsize = 16)
    ax.set_xlabel("Cluster", fontsize = 18)
    ax.set_ylabel("Mean Standardized Value", fontsize = 18)
    ax.tick_params(axis = "y", labelsize = 16)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.5)

    # create custom legend with full opacity
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor = color_map[f], label = f) for f in analysis_cols]
    ax.legend(handles = legend_handles, fontsize = 14)

    plt.tight_layout()
    plt.show()

    return tox_norm_centroids
    
    
    
def plot_toxicity_boxplots(real_df, sim_df, tox_cols, name_1='real', name_2='simulated', cluster_col=None, cluster_id=None):
    # filter by cluster
    if cluster_col and cluster_id is not None:
        real_df = real_df[real_df[cluster_col] == cluster_id]
        sim_df = sim_df[sim_df[cluster_col] == cluster_id]

    # prep data for plot
    real_melted = real_df[tox_cols].melt(var_name='trait', value_name='value')
    real_melted['dataset'] = name_1

    sim_melted = sim_df[tox_cols].melt(var_name='trait', value_name='value')
    sim_melted['dataset'] = name_2

    plot_df = pd.concat([real_melted, sim_melted], axis=0)

    # plot
    plt.figure(figsize=(10, 6))
    plt.yscale("log")
    sns.boxplot(data=plot_df, x='trait', y='value', hue='dataset')
    plt.title(f'Toxicity Feature Comparison{" - Cluster " + str(cluster_id) if cluster_id is not None else ""}')
    plt.ylabel('Toxicity Score')
    plt.xlabel('Toxicity Trait')
    plt.ylim(0, 1)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()
    
    
    
def plot_toxicity_histograms(real_df, sim_df, tox_cols, name_1 = 'Real', name_2 = 'Simulated', bins = 50, kde = True, figsize = (12, 8), sharey = True, transform = None):
    def _apply_transform(x, method):
        if method == 'log1p':
            return np.log1p(x)
        elif method == 'sqrt':
            return np.sqrt(x)
        elif method == 'logit':
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))
        return x  # no transform

    n_traits = len(tox_cols)
    ncols = 2
    nrows = (n_traits + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=sharey)
    axes = axes.flatten()

    for i, trait in enumerate(tox_cols):
        ax = axes[i]

        real_vals = _apply_transform(real_df[trait].dropna(), transform)
        sim_vals = _apply_transform(sim_df[trait].dropna(), transform)

        sns.histplot(real_vals, bins=bins, kde=kde, stat='density', label=name_1,
                     color='steelblue', alpha=0.6, ax=ax)
        sns.histplot(sim_vals, bins=bins, kde=kde, stat='density', label=name_2,
                     color='darkorange', alpha=0.6, ax=ax)

        ax.set_title(f"{trait} ({'transformed' if transform else 'raw'})")
        ax.set_xlabel('Transformed Score' if transform else 'Score')
        ax.set_ylabel('Density')
        ax.legend()

    # remove unused subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Real vs Simulated Toxicity Trait Distributions', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()