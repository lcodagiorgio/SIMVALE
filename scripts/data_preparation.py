##### Base libs
import pandas as pd
from collections import Counter
from tqdm import tqdm
from random import seed, randint

##### Cleaning and time handling libs 
# time analysis 
from datetime import datetime
# text cleaning
from html import unescape
import re
import string
import contractions

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)



# ------------------- DATA PREPARATION functions -------------------



def convert_utc_time(df, utc_time_col):
    # convert the utc timestamp into date and time of day features
    # convert utc to datetime
    df["datetime"] = pd.to_datetime(df[utc_time_col], unit = "s")
    # extract date as string (YYYY-MM-DD)
    df["date"] = df["datetime"].dt.date.astype(str)
    # extract time as string (HH:MM:SS)
    df["time_of_day"] = df["datetime"].dt.time.astype(str)

    # drop original and temporary features
    df.drop(columns = [utc_time_col, "datetime"], inplace = True)



def analyze_distinct(df):
    # distinct values
    to_analyze = []

    print("Number of distinct values:")
    for c in df.columns:
        uniques = df[c].unique()
        print(f"{c}: {len(uniques)}")
        if len(uniques) < 20:
            to_analyze.append((c, uniques))

    # show distinct values
    print("\nDistinct values:")
    for c, dist in to_analyze:
        print(f"{c}:\n{dist}")
        


def process_dataset(df, text_col):
    # remove duplicate comments
    df.drop_duplicates(subset = [text_col], inplace = True)
    print(f"Shape after duplicates removal:\n{df.shape}")

    # checking for empty comments
    print("Check for empty comments")
    print(len(df[df[text_col].isin(["", " ", "NaN", "None", "NULL", "null", "NA"])]))
    print("Check for NaN comments")
    print(len(df[df[text_col].isna()]))

    # drop NaN comments
    df = df[df[text_col].notna()].reset_index(drop = True)
    print(f"Shape:\n{df.shape}\n")



def check_empty(df, col):
    # checking for empty str comments after cleaning
    print(f"Empty:\n{len(df[df[col] == ''])}")
    # removing empty comments and resetting index
    df = df[df[col] != ""].reset_index(drop = True)
        
    

def general_text_cleaning(df, text_col):
    def _clean(text):
        if not isinstance(text, str):
            return ""
        # decode HTML entities
        text = unescape(text)
        # remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # remove mentions (@username) and hashtags (#topic)
        text = re.sub(r'@\w+|#\w+', "", text)
        # remove replacement symbol (erroneously decoded unicode chars)
        text = re.sub(r"�", "", text)
        # expand contractions
        text = contractions.fix(text, slang = True)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # remove leading and trailing spaces
        text = text.strip()
        
        return text
    
    # clean the comments' content column
    df[text_col] = df[text_col].progress_apply(_clean)
    
    print("Text cleaned.\n")