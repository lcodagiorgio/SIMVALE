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

##### Distances and similarity metrics
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

##### Distributions
from scipy.stats import ks_2samp, wasserstein_distance, entropy

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

##### Seed to ensure reproducibility
SEED = 42

np.random.seed(SEED)
seed(SEED)



# ------------------- FEATURE ENGINEERING functions -------------------



def feat_engineering(df):
    ## SENTIMENT
    # emotion balance
    df["sent_balance"] = df["positive"] - df["negative"]
    df["sent_emoji_balance"] = df["num_emoji_pos"] - df["num_emoji_neg"]
    ## LINGUISTIC
    # per word ratios
    df["punct_ratio"] = df["num_punct"] / df["num_words"]
    df["upper_ratio"] = df["num_words_upp"] / df["num_words"]
    df["emoji_ratio"] = df["num_emoji"] / df["num_words"]
    df["adj_ratio"] = df["num_words_adj"] / df["num_words"]
    df["noun_ratio"] = df["num_words_noun"] / df["num_words"]
    df["verb_ratio"] = df["num_words_verb"] / df["num_words"]
    df["lex_ratio"] = df["num_words_lex"] / df["num_words"]
    df["stopw_ratio"] = df["num_stopw"] / df["num_words"]
    # words per sentence ratio
    df["words_sent_ratio"] = df["num_words"] / df["num_sents"]
    # Type-Token Ratio (TTR) (number of unique tokens / number of total tokens) on LEMMATIZED text
    df["ttr"] = df["std_body"].progress_apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
    ## READABILITY
    # complexity ratio
    df["complex_ratio"] = df["difficult_words"] / df["num_words"]


# ------------------- VALIDATION functions -------------------



def match_clusters(df_centroids_1, df_centroids_2, name_1, name_2):
    # cosine similarity
    cos_sim = cosine_similarity(df_centroids_1.values, df_centroids_2.values)
    # cosine distance
    cos_dist = 1 - cos_sim

    # linear sum assignment algorithm 
    # (to pair each df_1 cluster to one df_2 cluster such that the overall cosine similarity is maximized)
    row_ind, col_ind = linear_sum_assignment(cos_dist)

    # matching clusters
    df_match = pd.DataFrame({f"{name_1} cluster": df_centroids_1.index[row_ind], 
                               f"Best {name_2} cluster": df_centroids_2.index[col_ind], 
                               "Cosine similarity": cos_sim[row_ind, col_ind]})

    # rename the index according to the analysis
    df_match.set_index(f"{name_1} cluster", inplace = True)
    df_similarity = pd.DataFrame(cos_sim, index = df_centroids_1.index, columns = df_centroids_2.index)
    
    return df_match, df_similarity



def annotate_profile(df, profile_map, cluster_col = "cluster", profile_name = "profile"):
    # label the clusters according to the profiles found
    df = df.copy()
    df[profile_name] = df[cluster_col].map(profile_map)
    
    return df



def significant_features(df_1, df_2, cols, pval_thr, ordinal = False):
    # list of significally different features according to KS test on distributions
    different_feat = []
    # non-significally different features
    equal_feat = []

    for feat in cols:
        if ordinal:
            # discrete ordinal feature
            num_map = {"very low": 1, "low": 2, "medium": 3, "high": 4, "very high": 5}
            df_1_map = df_1[feat].map(num_map)
            df_2_map = df_2[feat].map(num_map)
            # KS test
            _, ks_pval = ks_2samp(df_1_map, df_2_map)
        else:
            # continuous feature
            # KS test
            _, ks_pval = ks_2samp(df_1[feat], df_2[feat])
        
        # keep only significally different features
        if ks_pval < pval_thr:
            different_feat.append(feat)
        else:
            equal_feat.append(feat)
            
    return equal_feat, different_feat



def eval_features(df_1, df_2, diff_cols):
    # store group metrics
    metrics_all = {}
    
    for trait in diff_cols:
        # store metrics for single trait
        metrics = {}
        
        # mean and std
        metrics["rel_mean_diff"] = (df_1[trait].mean() - df_2[trait].mean()) / df_1[trait].mean() if df_1[trait].mean() != 0 else None
        metrics["rel_std_diff"] = (df_1[trait].std() - df_2[trait].std()) / df_1[trait].std() if df_1[trait].std() != 0 else None
        
        # percentiles
        perc = [10, 25, 50, 75, 90]
        df_1_perc = np.percentile(df_1[trait], perc)
        df_2_perc = np.percentile(df_2[trait], perc)
        for i, p in enumerate(perc):
            metrics[f"rel_p{p}_diff"] = (df_1_perc[i] - df_2_perc[i]) / df_1_perc[i] if df_1_perc[i] != 0 else None
        
        # Kolmogorov-Smirnov distance to compare distribution shapes
        ks_stat, _ = ks_2samp(df_1[trait], df_2[trait])
        metrics["KS"] = ks_stat
        
        # Earth Mover Distance to compute the cost of transforming one distribution into the other
        #metrics["EMD"] = wasserstein_distance(df_1[trait], df_2[trait])

        # update all metrics for the trait
        metrics_all[trait] = metrics
        
    # round metrics for better readability
    for trait in metrics_all:
        for key, value in metrics_all[trait].items():
            if isinstance(value, dict):
                try:
                    metrics_all[trait][key] = {k: round(v, 5) for k, v in value.items()}
                except:
                    pass
            elif isinstance(value, float):
                metrics_all[trait][key] = round(value, 5)
        
    return metrics_all



def eval_personality(df_1, df_2, pers_cols):
    # store personality metrics
    metrics_pers = {}
    
    for trait in pers_cols:
        # store metrics for single trait
        metrics = {}
        
        # proportions of different levels
        levels = ["very low", "low", "medium", "high", "very high"]
        df_1_prop = pd.Series(pd.Categorical(df_1[trait], categories = levels)).value_counts(normalize = True).sort_index()
        df_2_prop = pd.Series(pd.Categorical(df_2[trait], categories = levels)).value_counts(normalize = True).sort_index()
        metrics["prop_diff"] = dict(df_1_prop - df_2_prop)
         
        # matching of the most common trait level 
        is_match = df_1_prop.idxmax() == df_2_prop.idxmax()
        metrics["dominant"] = {"match": is_match, "df_1": df_1_prop.idxmax(), "df_2": df_2_prop.idxmax()}
        
        # numerical mapping
        num_map = {"very low": 1, "low": 2, "medium": 3, "high": 4, "very high": 5}
        df_1_map = df_1[trait].map(num_map)
        df_2_map = df_2[trait].map(num_map)
        
        # entropy
        metrics["entropy"] = {"df_1": entropy(df_1_prop, base = 2), "df_2": entropy(df_2_prop, base = 2), 
                              "diff": (entropy(df_1_prop, base = 2) - entropy(df_2_prop, base = 2)) / entropy(df_1_prop, base = 2)}
        
        # Earth Mover Distance to compute the cost of transforming one distribution into the other
        #metrics["EMD"] = wasserstein_distance(df_1_map, df_2_map)
        
        # update the personality metrics for the trait
        metrics_pers[trait] = metrics
    
    # round metrics for better readability
    for trait in metrics_pers:
        for key, value in metrics_pers[trait].items():
            if isinstance(value, dict):
                try:
                    metrics_pers[trait][key] = {k: round(v, 5) for k, v in value.items()}
                except:
                    pass
            elif isinstance(value, float):
                metrics_pers[trait][key] = round(value, 5)

    return metrics_pers



def print_metrics(metrics_dict, sort_by, top):
    # sort by metric
    metrics_sort = sorted(metrics_dict.items(), key = lambda x: np.abs(x[1].get(sort_by, 0)), reverse = True)
    
    if not top:
        print("\nMetrics for all significally different features:")
        top = len(metrics_dict)
    else:
        print(f"\nMetrics for top {top} significally different features according to {sort_by}:")
    for feat, stats in metrics_sort[:top]:
        print(f"** {feat} **")
        for key, val in stats.items():
            if isinstance(val, dict):
                print(f"\t{key}:")
                for sub_key, sub_val in val.items():
                    print(f"\t- {sub_key}: {sub_val}")
            else:
                print(f"\t{key}: {val}")
                


def validation_metrics(df_1, df_2, tox_cols, pers_cols, sent_cols, ling_cols, read_cols, 
                       pval_thr = 0.05, sort_by = "rel_mean_diff", top = None, profile = None, 
                       name_1 = "Real", name_2 = "Simulated"):
    # function to compare two datasets' features across multiple aspects
    # if profile is None select whole dataset
    # otherwise select the profile to analyze
    if profile:
        df_1 = df_1[df_1["tox_profile"] == profile]
        df_2 = df_2[df_2["tox_profile"] == profile]

    print(f"## Validating {name_1} against {name_2} data ##")
    if profile:
        print(f"LOCAL VALIDATION: {profile} profile")
        print("----------------------\n")
    else:
        print(f"GLOBAL VALIDATION")
        print("----------------------\n")
        
    # store all non-different features
    equal_feats = {}
    # store all different features
    diff_feats = {}
    # store all validation metrics
    val_metrics = {}
    
    print(f"Size of {name_1} data: {len(df_1)}\nSize of {name_2} data: {len(df_2)}")
    
    print(f"\n\n------- TOXIC dimension -------")
    # retrieve significally different features based on KS test on distributions
    equal_tox, different_tox = significant_features(df_1, df_2, tox_cols, pval_thr = pval_thr)
    # compute metrics
    metrics_tox = eval_features(df_1, df_2, different_tox)
    equal_feats["TOXICITY"] = equal_tox
    diff_feats["TOXICITY"] = different_tox
    val_metrics["TOXICITY"] = metrics_tox
    print(f"Non-significally different features (p-value < {pval_thr}):\n{equal_tox}")
    print(f"Significally different features (p-value < {pval_thr}):\n{different_tox}")
    print_metrics(metrics_tox, sort_by = sort_by, top = top)
    
    
    print(f"\n\n------- PERSONALITY dimension -------")
    # retrieve significally different features based on KS test on distributions
    equal_pers, different_pers = significant_features(df_1, df_2, pers_cols, pval_thr = pval_thr, ordinal = True)
    # compute metrics
    metrics_pers = eval_personality(df_1, df_2, different_pers)
    equal_feats["PERSONALITY"] = equal_pers
    diff_feats["PERSONALITY"] = different_pers
    val_metrics["PERSONALITY"] = metrics_pers
    print(f"Non-significally different features (p-value < {pval_thr}):\n{equal_pers}")
    print(f"Significally different features (p-value < {pval_thr}):\n{different_pers}")
    print_metrics(metrics_pers, sort_by = sort_by, top = None)
    
    
    print(f"\n\n------- SENTIMENT/EMOTIONAL dimension -------")
    # retrieve significally different features based on KS test on distributions
    equal_sent, different_sent = significant_features(df_1, df_2, sent_cols, pval_thr = pval_thr)
    # compute metrics
    metrics_sent = eval_features(df_1, df_2, different_sent)
    equal_feats["SENTIMENT"] = equal_sent
    diff_feats["SENTIMENT"] = different_sent
    val_metrics["SENTIMENT"] = metrics_sent
    print(f"Non-significally different features (p-value < {pval_thr}):\n{equal_sent}")
    print(f"Significally different features (p-value < {pval_thr}):\n{different_sent}")
    print_metrics(metrics_sent, sort_by = sort_by, top = top)
    
    
    print(f"\n\n------- LINGUISTIC dimension -------")
    # retrieve significally different features based on KS test on distributions
    equal_ling, different_ling = significant_features(df_1, df_2, ling_cols, pval_thr = pval_thr)
    # compute metrics
    metrics_ling = eval_features(df_1, df_2, different_ling)
    equal_feats["LINGUISTIC"] = equal_ling
    diff_feats["LINGUISTIC"] = different_ling
    val_metrics["LINGUISTIC"] = metrics_ling
    print(f"Non-significally different features (p-value < {pval_thr}):\n{equal_ling}")
    print(f"Significally different features (p-value < {pval_thr}):\n{different_ling}")
    print_metrics(metrics_ling, sort_by = sort_by, top = top)
    
    
    print(f"\n\n------- READABILITY dimension -------")
    # retrieve significally different features based on KS test on distributions
    equal_read, different_read = significant_features(df_1, df_2, read_cols, pval_thr = pval_thr)
    # compute metrics
    metrics_read = eval_features(df_1, df_2, different_read)
    equal_feats["READABILITY"] = equal_read
    diff_feats["READABILITY"] = different_read
    val_metrics["READABILITY"] = metrics_read
    print(f"Non-significally different features (p-value < {pval_thr}):\n{equal_read}")
    print(f"Significally different features (p-value < {pval_thr}):\n{different_read}")
    print_metrics(metrics_read, sort_by = sort_by, top = top)
    
    
    return equal_feats, diff_feats, val_metrics



def validation_metrics_mod(df_1, df_2, tox_cols, pers_cols, sent_cols, ling_cols, read_cols, 
                       pval_thr = 0.05, sort_by = "rel_mean_diff", top = None, profile = None, 
                       name_1 = "Unmoderated", name_2 = "Moderated"):
    # function to compare two datasets' features across toxicity dimension
    # if profile is None select whole dataset
    # otherwise select the profile to analyze
    if profile:
        df_1 = df_1[df_1["tox_profile"] == profile]
        df_2 = df_2[df_2["tox_profile"] == profile]

    print(f"## Validating {name_1} against {name_2} data ##")
    if profile:
        print(f"LOCAL VALIDATION: {profile} profile")
        print("----------------------\n")
    else:
        print(f"GLOBAL VALIDATION")
        print("----------------------\n")
        
    # store all non-different features
    equal_feats = {}
    # store all different features
    diff_feats = {}
    # store all validation metrics
    val_metrics = {}
    
    print(f"Size of {name_1} data: {len(df_1)}\nSize of {name_2} data: {len(df_2)}")
    
    print(f"\n\n------- TOXIC dimension -------")
    # retrieve significally different features based on KS test on distributions
    equal_tox, different_tox = significant_features(df_1, df_2, tox_cols, pval_thr = pval_thr)
    # compute metrics
    metrics_tox = eval_features(df_1, df_2, different_tox)
    equal_feats["TOXICITY"] = equal_tox
    diff_feats["TOXICITY"] = different_tox
    val_metrics["TOXICITY"] = metrics_tox
    print(f"Non-significally different features (p-value < {pval_thr}):\n{equal_tox}")
    print(f"Significally different features (p-value < {pval_thr}):\n{different_tox}")
    print_metrics(metrics_tox, sort_by = sort_by, top = top)
    
    return equal_feats, diff_feats, val_metrics



#def compute_goodness_scores(metrics_dict, thr = 0.2):
#    scores = {}
#    for dimension, features in metrics_dict.items():
#        total = 0
#        good = 0
#        for _, stats in features.items():
#            total += 1
#            if 'rel_mean_diff' in stats:
#                if abs(stats['rel_mean_diff']) < thr:
#                    good += 1
#            elif 'entropy' in stats:
#                entropy_ok = abs(stats['entropy']['diff']) < thr
#                if entropy_ok:
#                    good += 1
#        scores[dimension] = round(good / total, 3) if total > 0 else None
#    return scores
#
#
#
#def plot_goodness(metrics_dict, thresholds):
#    from matplotlib import cm
#    # Define distinct colors for each threshold combination
#    colors = [cm.magma(i) for i in np.linspace(0.2, 0.9, len(thresholds))]
#    labels = list(metrics_dict.keys())
#    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#    angles += angles[:1]
#
#    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
#
#    if not colors:
#        colors = plt.cm.Greys(np.linspace(0.4, 1.0, len(thresholds)))
#
#    for thr, color in zip(thresholds, colors):
#        scores = compute_goodness_scores(metrics_dict, thr)
#        values = list(scores.values())
#        values += values[:1]
#
#        ax.plot(angles, values, color=color, linewidth=2, label=f"thr = {int(thr*100)}%")
#        ax.fill(angles, values, color=color, alpha=0.2)
#
#    # Layout
#    ax.set_theta_offset(np.pi / 2)
#    ax.set_theta_direction(-1)
#    for angle in angles[:-1]:
#        ax.plot([angle, angle], [0, 1], color='black', linewidth=0.8, linestyle='dotted')
#
#    label_radius = 1.08
#    for angle, label in zip(angles[:-1], labels):
#        ax.text(angle, label_radius, label, fontsize=16, color='black',
#                ha='center', va='center',
#                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))
#
#    ax.set_ylim(0, 1)
#    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
#    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color='black', fontsize=14)
#    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.8)
#    ax.xaxis.grid(False)
#    ax.spines['polar'].set_color('black')
#    ax.spines['polar'].set_linewidth(1.5)
#    ax.set_xticks([])
#
#    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=12, title = "Relative difference threshold", title_fontsize = 12)
#    plt.tight_layout()
#    plt.show()