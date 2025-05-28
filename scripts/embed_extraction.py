##### Base libs
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import seed, randint
from collections import Counter
import matplotlib.pyplot as plt 
import seaborn as sns

##### Shuffling and resampling 
from sklearn.utils import shuffle, resample

##### Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

##### Dimensionality reduction
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

##### LLMs libs
# pytorch (with CUDA)
import torch
from torch.optim import AdamW
# learning rate scheduler
from transformers import get_linear_schedule_with_warmup
# tokenizer and configuration
from transformers import AutoTokenizer, AutoConfig
# model with classification head (fine-tuning purpose) and optimizer
from transformers import AutoModelForSequenceClassification
# data loader to batch and use pytorch tensors
from torch.utils.data import DataLoader, Dataset
# label encoder
from sklearn.preprocessing import LabelEncoder
# train-test split
from sklearn.model_selection import train_test_split
# evaluation of fine-tuned model
from sklearn.metrics import f1_score

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

##### Seed to ensure reproducibility
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
seed(SEED)
torch.cuda.manual_seed_all(SEED)



# ------------------- FINE-TUNING classes and functions -------------------



def ft_data(df, label_col, imb_ratio = 3, test_size = 0.1):
    print(f"Processing {label_col.capitalize()} data...\n")
    print(f"Original class distribution: {Counter(df[label_col])}")

    # minority class
    class_counts = df[label_col].value_counts()
    min_count = class_counts.min()

    # resample each class to balance
    balanced_dfs = []
    for label, count in class_counts.items():
        sample_n = min(imb_ratio * min_count, count)
        df_class = df[df[label_col] == label].sample(n = sample_n, random_state = SEED)
        balanced_dfs.append(df_class)

    # combine data
    df_ft = pd.concat(balanced_dfs).reset_index(drop = True)
    print(f"More balanced class distribution: {Counter(df_ft[label_col])}")

    print(f"Splitting into FT and test set...")
    df_ft, df_test = train_test_split(df_ft, test_size = test_size, stratify = df_ft[label_col], random_state = SEED)

    print(f"Shape of data for FT: {df_ft.shape}")
    print(f"Shape of the test data: {df_test.shape}")
    
    return df_ft, df_test



class BertDataset(Dataset):
    # class needed to use data efficiently with pytorch models
    # perform tokenization and format labels
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = 256

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], 
                                  padding = "max_length", 
                                  truncation = True, 
                                  max_length = self.max_len, 
                                  return_tensors = "pt")
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype = torch.long)
        
        return item



def prepare_data(df, text_col, label_col, sample_size, val_size, balance = True):
    # original class distribution
    print(f"Original class distribution:\n{Counter(df[label_col])}")
    
    # label encoding
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df[label_col])
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    num_labels = len(label_mapping)
    print(f"Number of labels: {num_labels}, Label Mapping: {label_mapping}")

    # sample size
    sample_size = min(sample_size, len(df))
    # define train size
    train_size = int(sample_size*(1 - val_size)) + 1
    val_size = sample_size - train_size

    # stratified sample as validation data
    df_val, _ = train_test_split(df, train_size = val_size, stratify = df["label"], random_state = SEED)
    val_set, val_lab = df_val[text_col].tolist(), df_val["label"].tolist()
    
    # avoid data leakage
    df_rem = df.drop(df_val.index)
    
    # balance sample of the training set or take stratified sample (according to the labels)
    if balance:
        print("\nSampling balanced training set...")
        per_class = train_size // num_labels
        # undersample majority classes
        df_train = df_rem.groupby(label_col, group_keys = False).apply(lambda x: x.sample(min(len(x), per_class), random_state = SEED))
        train_set, train_lab = df_train[text_col].tolist(), df_train["label"].tolist()
    else:
        df_train, _ = train_test_split(df_rem, train_size = train_size, stratify = df_rem["label"], random_state = SEED)
        train_set, train_lab = df_train[text_col].tolist(), df_train["label"].tolist()

    print(f"\nTraining set size: {len(train_set)}")
    print(f"Training set class distribution:\n{Counter(train_lab)}\n")

    print(f"\nValidation set size: {len(val_set)}")
    print(f"Validation set class distribution:\n{Counter(val_lab)}\n")
    
    return train_set, train_lab, val_set, val_lab, label_mapping



def eval_model(model, data_loader, device):
    # evaluation mode
    model.to(device)
    model.eval()
    
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            # move inputs to GPU
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch["labels"].to(device)
            # inference
            outputs = model(**inputs, labels = labels)

            # extract loss
            loss = outputs.loss
            total_loss += loss.item()

            # val preds
            preds = torch.argmax(outputs.logits, dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss/len(data_loader)
    f1 = f1_score(all_labels, all_preds, average = "macro")

    return avg_loss, f1



def finetune_bert(df, text_col, label_col, exp_name, bert_variant = "bert-base-uncased", epochs = 5, lr = 3e-5, dropout = 0.2,
                  n_freeze = 0, sample_size = 10000, val_size = 0.1, batch_size = 16, stop_early = False, balance = True):
    # prepare data
    train_set, train_lab, test_set, test_lab, label_mapping = prepare_data(df, text_col, label_col, sample_size, val_size, balance = balance)
    num_labels = len(label_mapping)
    print(f"\nFine-tuning BERT on {exp_name.capitalize()}\n")
    
    # tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(bert_variant)
    train_data = BertDataset(train_set, train_lab, tokenizer)
    val_data = BertDataset(test_set, test_lab, tokenizer)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)
    
    # model configuration
    config = AutoConfig.from_pretrained(bert_variant, num_labels = num_labels, hidden_dropout_prob = dropout, attention_probs_dropout_prob = dropout)
    model = AutoModelForSequenceClassification.from_pretrained(bert_variant, config = config)
    
    # freeze embedding layer and some transformer layers
    if n_freeze != 0:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"Frozen embeddings and first {n_freeze} layers, training only the top {12-n_freeze} layers.")
    
    # optimizer and learning rate schedul
    optimizer = AdamW(model.parameters(), lr = lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(0.1 * total_steps), 
                                                num_training_steps = total_steps)
    
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # track losses and performances across epochs
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    
    # early stopping params
    best_f1 = 0
    patience_counter = 0

    # training loop
    for epoch in range(epochs):
        # training mode
        model.train()
        loop = tqdm(train_loader, desc = f"Epoch {epoch + 1}/{epochs}", leave = True)
        # losses, predictions and labels per batch
        total_train_loss = 0
        all_preds, all_labels = [], []
        
        for batch in loop:
            # reset gradients to not accumulate them across batches
            optimizer.zero_grad()
            
            # move inputs to GPU
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch["labels"].to(device)
            # inference
            outputs = model(**inputs, labels = labels)
            
            # extract loss
            loss = outputs.loss
            # backprop
            loss.backward()
            # update params and lr
            optimizer.step()
            scheduler.step()
            
            # train loss
            total_train_loss += loss.item()
            # train preds
            preds = torch.argmax(outputs.logits, dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # show batch loss
            loop.set_postfix(loss = loss.item())

        # avg training loss and f1 for epoch
        avg_train_loss = total_train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        train_f1 = f1_score(all_labels, all_preds, average = "macro")
        train_f1s.append(train_f1)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.5f}, Train F1: {train_f1:.5f}")

        # avg validation loss and f1
        model.eval()
        val_loss, val_f1 = eval_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.5f}, Val F1: {val_f1:.5f}\n")
        
        # early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # save the best model
            model.save_pretrained(f"../models_tox/bert_ft_{exp_name}_best")
            tokenizer.save_pretrained(f"../models_tox/bert_ft_{exp_name}_best")
        else:
            patience_counter += 1
            
        if stop_early and patience_counter >= 2:
            print(f"Early stopping at epoch {epoch + 1}. Best F1: {best_f1:.5f}")
            break

    # save fine-tuned model and tokenizer
    model.save_pretrained(f"../models_tox/bert_ft_{exp_name}")
    tokenizer.save_pretrained(f"../models_tox/bert_ft_{exp_name}")

    print(f"Fine-tuned model saved for {exp_name}\n")
    
    return train_losses, val_losses, train_f1s, val_f1s



def init_bert(trait_name):
    # load BERT best tokenizer and model fine-tuned on trait
    path = f"../models_tox/bert_ft_{trait_name}_best"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, output_hidden_states = True)

    # set model for inference
    model.eval()
    # move model to GPU
    model.to("cuda")

    return tokenizer, model



def inference_bert(df_test, text_col, label_col, exp_name, batch_size = 16):
    # load tokenizer and model
    tokenizer, model = init_bert(exp_name)
    
    # label encoding
    encoder = LabelEncoder()
    df_test["label"] = encoder.fit_transform(df_test[label_col])
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    num_labels = len(label_mapping)
    print(f"Number of labels: {num_labels}, Label Mapping: {label_mapping}")
    
    # test data
    test_set, test_lab = df_test[text_col].tolist(), df_test["label"].tolist()
    print(f"Test set size: {len(test_set)}")
    print(f"Test set class distribution:\n{Counter(test_lab)}\n")
    
    test_data = BertDataset(test_set, test_lab, tokenizer)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
    
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_loss, test_f1 = eval_model(model, test_loader, device)
    
    print(f"Test Loss: {test_loss:.5f}, Test F1: {test_f1:.5f}\n")
    
    return test_loss, test_f1



def plot_learn_curve(train_loss, val_loss, train_perf, val_perf, trait_name, perf_name):
    sns.set(style = "whitegrid", context = "talk")
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize = (11, 5))

    # loss axis (left)
    ax1.plot(epochs, train_loss, marker = 'o', linestyle = '-', color = 'darkred', linewidth = 2, label = "Training Avg Loss")
    ax1.plot(epochs, val_loss, marker = 'X', linestyle = '--', color = 'orange', linewidth = 2, label = "Validation Avg Loss")
    ax1.set_xlabel("Epoch", fontsize = 24)
    ax1.set_ylabel("Average Loss", color = 'darkred', fontsize = 24)
    ax1.tick_params(axis = 'y', labelcolor = 'darkred', labelsize = 22)
    ax1.tick_params(axis = 'x', labelsize = 22)

    # performance axis (right)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_perf, marker = 'o', linestyle = '-', color = 'darkblue', linewidth = 2, label = f"Training Macro-{perf_name}")
    ax2.plot(epochs, val_perf, marker = 'X', linestyle = '--', color = 'royalblue', linewidth = 2, label = f"Validation Macro-{perf_name}")
    ax2.set_ylabel(f"Macro-{perf_name}", color = 'darkblue', fontsize = 24)
    ax2.tick_params(axis = 'y', labelcolor = 'darkblue', labelsize = 22)

    ax1.grid(True, linestyle = "--", alpha = 0.5)
    fig.tight_layout()
    #fig.subplots_adjust(top = 0.95)

    # combine legends
    #lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(
    #    lines1 + lines2,
    #    labels1 + labels2,
    #    loc = "lower center",
    #    ncol = 2,
    #    bbox_to_anchor = (0.5, 1),
    #    fontsize = 20,
    #    frameon = False
    #)

    plt.show()



# ------------------- EMBEDDINGS EXTRACTION classes and functions -------------------



def extract_embeddings(df, text_col, traits, pooling = "cls"):
    def _get_embedding(text, tokenizer, model, pooling):
        inputs = tokenizer(text, return_tensors = "pt", truncation = True, padding = True, max_length = 256)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            # inference
            outputs = model(**inputs)
        
        # extract last hidden layer
        last_hidden_state = outputs.hidden_states[-1]
        
        if pooling == "cls":
            # return [CLS] embedding (whole text)
            return last_hidden_state[:, 0, :]
        elif pooling == "mean":
            # mean aggregation of embeddings
            return last_hidden_state.mean(dim = 1)
        else:
            print("Pooling must be cls or mean.")
            exit()

    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # scaling the embeddings from each model before concat, one may dominate
    scaler = StandardScaler()
    
    # init full embedding list to then concat
    all_embeddings = []
    # iterate over traits to extract related embeddings
    for trait in traits:
        print(f"\nExtracting embeddings for {trait.capitalize()}")
        tokenizer, model = init_bert(trait_name = trait)
        embeddings = df[text_col].progress_apply(lambda x: _get_embedding(x, tokenizer, model, pooling))
        embeddings = torch.cat(embeddings.tolist(), dim = 0).cpu().numpy()
        # scale
        embeddings_std = scaler.fit_transform(embeddings)
        # add to list of embs to concatenate
        all_embeddings.append(embeddings_std)

    # concat embeddings from all fine-tuned models
    final_embeddings = torch.cat([torch.tensor(emb) for emb in all_embeddings], dim = -1).numpy()
    
    return final_embeddings



def minmax_scale(values):
    # minmax normalization
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)
    
    # return scaled values and the fitted scaler
    return values_scaled, scaler



def zscore_scale(values):
    # zscore standardization
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)
    
    # return scaled values and the fitted scaler
    return values_scaled, scaler



def apply_pca(values, explain_perc = 0.75):
    # apply PCA on embeddings
    pca = PCA(n_components = explain_perc)
    values_pca = pca.fit_transform(values)

    print(f"Shape before PCA: {values.shape}")
    print(f"Shape after PCA: {values_pca.shape}")

    # cumulative explained variance
    expl_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = len(expl_var_ratio)

    # style
    sns.set(style = "whitegrid", context = "talk")

    # plot
    plt.figure(figsize = (10, 5))
    plt.plot(
        range(1, num_components + 1),
        expl_var_ratio,
        marker = "o",
        linestyle = "--",
        color = "darkblue",
        linewidth = 2,
        markersize = 6
    )

    # labels and formatting
    plt.xlabel("Number of Components", fontsize = 20)
    plt.ylabel("Cumulative Explained Variance", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.tight_layout()
    plt.show()

    return values_pca, pca



def add_comment_id(df, embeddings):
    # extract comment ids from dataframe
    comment_ids = df["comment_id"].values.reshape(-1, 1)
    # add comment ids to embeddings
    embs = np.hstack((comment_ids, embeddings))
    
    return embs