##### Base libs
import pandas as pd
from collections import Counter
from tqdm import tqdm
from random import seed, randint
import matplotlib.pyplot as plt

##### Vectorization and text analysis libs
# vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# wordcloud
from wordcloud import WordCloud
import nltk  
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

##### Toxicity detection
# detoxify
from detoxify import Detoxify
# Perspective API
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)







# ------------------- TOXICITY DETECTION AND VECTORIZATION functions -------------------



def extract_toxicity(df, text_col, model_name = "original"):
    # extract toxicity features from the comments
    def _get_toxicity(text):
        toxicity_scores = tox_model.predict(text)
        
        return pd.Series(toxicity_scores)
    
    tox_model = Detoxify(model_name, device = "cuda")
    
    df[["toxicity", "severe_toxicity", "obscene", 
        "threat", "insult", "identity_attack"]] = df[text_col].progress_apply(_get_toxicity)

    print("Toxicity features extracted.\n")



def toxic_labeling(df, tox_col, thr = 0.7):
    # annotating comment with toxic or not
    df["is_toxic"] = df[tox_col].progress_apply(lambda x: 1 if x >= thr else 0)




def word_cloud(df, text_col, analysis_col, top = 10000, max_words = 200, stopwords = None, color = None):
    # no stopwords to remove
    if stopwords is None:
        stopwords = set()
        
    # top comments based on the feature to inspect
    df_top = df.nlargest(top, analysis_col)

    # extract top comments and concatenate them
    text_data = " ".join(set(df_top[text_col]))

    # word cloud
    word_cloud = WordCloud(max_font_size = 100, 
                           max_words = max_words,
                           stopwords = stopwords,
                           background_color = "white",
                           font_path = "../auxiliary_data/helvetica.ttf",
                           colormap = color).generate(text_data)

    # extract the most frequent terms in the word cloud based on the feature analyzed (type of toxicity)
    top_n = 30
    prominent_words = set(sorted(word_cloud.words_, key = word_cloud.words_.get, reverse = True)[:top_n])
    # remove outlier terms
    prominent_words -= set(["fffd fffd", "fuck fuck"])
    
    # show the word cloud
    plt.figure(figsize = (10, 4))
    plt.title(f"{analysis_col.replace('_', ' ').title()}", fontsize = 15)
    plt.imshow(word_cloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.show()

    return prominent_words
    


def toxic_tfidf_vectorize(df, text_col, toxic_terms):
    # initialize the tf-idf vectorizer focused on toxic terms
    vectorizer = TfidfVectorizer(vocabulary = toxic_terms)
    # compute tf-idf for the data
    tfidf = vectorizer.fit_transform(df[text_col])
    # toxicity-related tf-idf dataframe
    df_tfidf = pd.DataFrame(tfidf.toarray(), columns = vectorizer.get_feature_names_out())
    # update the original df with tf-idf features
    for term in toxic_terms:
        df[term] = df_tfidf[term]
