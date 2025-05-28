##### Base libs
import pandas as pd
from collections import Counter
from tqdm import tqdm
from random import seed, randint

##### NLP libs
# emoji analysis
import emoji
import unicodedata 
# time analysis
from datetime import datetime 
# text cleaning
from html import unescape
import re
import string
import contractions
# tokenizers, pos taggers, lemmatizers
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import stanza
#import spacy
# emotion detection
from nrclex import NRCLex
# polarity extraction
from textblob import TextBlob
# readability analysis
import textstat

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)



# ------------------- FEATURE EXTRACTION data objects -------------------



EMOTICON_MAP = {"positive": [":)", ":-)", ":')", ":]", ":-]", ":3", ":-3", ":>", ":->", "8)", "8-)", ":}", ":-}",
                                ":o)", "xD", ":o]", ":o3", ":o>", ":o}", ":c)", ":c]", ":c3", ":c>", ":c}",
                                ":^)", ":^]", ":^3", ":^>", ":^}", "=^)", "=^]", "=^3", "=^>", "=^}",
                                "=)", "=-)", "=]", "=-]", "=3", "=-3", "=>", "=->", "=}", "=-}",
                                "(:", "(-:", "[:", "[-:", "Ɛ:", "Ɛ-:", "<:", "<-:", "{:", "{-:",
                                ":D", ":-D", "8D", "8-D", "=D", "=-D", ":^D", "8^D", "=^D", "BD", "B-D", "B^D",
                                "^^", "^ ^", "^-^", "^_^", "^.^", "^,^", "^u^", "^w^", "^ㅂ^", "^o^", "^U^", "^O^", "^0^",
                                "(^_^)", "(^ ^)", "(^O^)", "(^o^)", "(^^)", "(≧∇≦)", "(◕ヮ◕)", "(^.^)", "(^·^)", "(*^0^*)",
                                "(●＾o＾●)", "(＾ｖ＾)", "(＾ｕ＾)", "(＾◇＾)", "( ^)o(^ )", "(^○^)", "＼(^o^)／", "\(^o^)/",
                                "(*^▽^*)", "(✿◠‿◠)", "ヽ(´ー｀)ﾉ", "（´∀｀）", "( ﾟヮﾟ)", "d(*⌒▽⌒*)b",
                                "(◕‿◕✿)", "(◡‿◡✿)", "(≧ ᗜ ≦)", "(⌃·̫⌃)", "٩(•̮̮̃•̃)۶", "٩(●̮̮̃●̃)۶", "٩(｡͡•‿•｡)۶",
                                "uwu", "UwU", "UuU", "UvU", "UvvU", "◕◡◕", "◕v◕", "◕u◕", "◕w◕",
                                "ツ", "シ", "ッ", "ヅ", "ツ゚", "ϡ", "ジ", "ｼ", "ﾂ", "㋡", "㋛", "☺", "☻",
                                "(*^▽^*)", "(✿◠‿◠)", "ヽ(´ー｀)ﾉ", "（´∀｀）", "( ﾟヮﾟ)", "d(*⌒▽⌒*)b",
                                "٩(•̮̮̃•̃)۶", "٩(●̮̮̃●̃)۶", "٩(｡͡•‿•｡)۶", "ლ(╹◡╹ლ)"],
                    
                    "negative": [":(", ":-(", ":[", ":'(", "-_-", ":-[", ":Ɛ", ":-Ɛ", ":<", ":-<", "8(", "8-(", ":{", ":-{",
                                ":o(", ":o[", ":oƐ", ":o<", ":o{", ":c(", ":c[", ":cƐ", ":c<", ":c{",
                                ":^(", ":^[", ":^Ɛ", ":^<", ":^{", "=^(", "=^[", "=^Ɛ", "=^<", "=^{",
                                "=(", "=-(", "=[", "=-[", "=Ɛ", "=-Ɛ", "=<", "=-<", "={", "=-{",
                                ":((", ":(((", ":((((", ":(((((", ":((((((", ":(((((((", ":((((((((",
                                "D:", "D-:", "Do:", "Dɔ:", "D^:", "D;", "D-;", "Do;", "Dɔ;", "D^;",
                                "ಠ_ಠ", "ლ(ಠ益ಠლ)", "ಠ╭╮ಠ", "ಠ▃ಠ", "ಠ益ಠ", "ಠ︵ಠ凸", "ಠ.ಠ", "ಠoಠ",
                                "ಥ_ಥ", "(╥﹏╥)", "(T_T)", "(;_;)", "(Ｔ▽Ｔ)", "(´；ω；`)", "(╥_╥)", "ಥ﹏ಥ",
                                "￣|○", "ಠ_ಠ", "ಠ益ಠ", "ಠ_ೃ", "ಠ_๏", "ಠ~ಠ", "ಥ_ಥ", "ಥ﹏ಥ", "ಥдಥ", 
                                "ಥ益ಥ", "ಠ︵ಠ", "ಠ‿ಠ", "ಠ⌣ಠ", "ಠ◡ಠ", "ಠ_ಥ", "ಥ_ಠ", "ಥ‿ಥ", "ಥ⌣ಥ", "ಥ﹏ಥ", "ಥ◡ಥ", "ಥ益ಥ"]
}

VAD = pd.read_csv("../auxiliary_data/VAD/NRC_EmoLex_NRC-VAD-Lexicon.txt", sep = "\t", header = None)

EMOJI_SENT = pd.read_csv("../auxiliary_data/emoji_sentiment/Emoji_Sentiment_Data_v1.0.csv", encoding = "utf-8")



# ------------------- FEATURE EXTRACTION functions -------------------



def extract_counts(df, text_col):
    # number of punctuation symbols
    df["num_punct"] = df[text_col].progress_apply(lambda x: sum(1 for char in x if char in string.punctuation)) 
    # number of sentences
    df["num_sents"] = df[text_col].progress_apply(lambda x: len(sent_tokenize(x)))
    # number of uppercase words (with length > 1, since we don't want to count e.g. "I", "A")
    df["num_words_upp"] = df[text_col].progress_apply(lambda x: sum(1 for w in word_tokenize(x) if w.isupper() and len(w) > 1))

    print("Counts of punctuation, sentences and uppercase words extracted.\n")



def extract_emoji_counts(df, text_col, emoji_sent = EMOJI_SENT, emoticon_map = EMOTICON_MAP):
    def _get_dominant(emoji_df):
        # extract dominant sentiment (we exclude neutral since dominates most of the scores)
        # we only look at score_pos <> score_neg to assign the dominant
        sent_scores = {"Positive": emoji_df["Positive"], "Negative": emoji_df["Negative"]}
        
        return max(sent_scores, key = sent_scores.get)
    
    def _count_emojis(text):
        # normalize unicode (NFKD removes variations)
        text = unicodedata.normalize("NFKD", text)
        # remove variation selectors, causing emoji inconsistencies
        text = text.replace("\ufe0f", "")
        
        # exclude copyright symbol (for successive encoding-decoding)
        list_emo = [e for e in emoji.emoji_list(text) if e["emoji"] != "©"]
        
        # list of decoded emojis
        emoji_names = [emoji.demojize(entry["emoji"]) for entry in list_emo]
        
        # extract emoticons (using the mapping)
        pos_emoticons = set(emoticon_map["positive"] + ["♡"])
        neg_emoticons = set(emoticon_map["negative"])
        # regex pattern to find all emoticons
        pattern = r"(" + r"|".join(map(re.escape, pos_emoticons.union(neg_emoticons))) + r")"

        # find all matching in text
        symb = re.findall(pattern, text)
        # number of emojis
        num_emoji = len(emoji_names + symb)
        # init counters
        num_emoji_pos = 0
        num_emoji_neg = 0

        # count each emoji-related sentiment
        for e in emoji_names + symb:
            sentiment = dict_emoji_sent.get(e)
            if sentiment == "Positive":
                num_emoji_pos += 1
            elif sentiment == "Negative":
                num_emoji_neg += 1

        return pd.Series({
            "num_emoji": num_emoji,
            "num_emoji_pos": num_emoji_pos,
            "num_emoji_neg": num_emoji_neg,
        })
    
    
    df_emoji_sent = emoji_sent
    
    # extract the textual representation for the emojis
    df_emoji_sent["demojized"] = df_emoji_sent["Emoji"].apply(emoji.demojize)

    # extract dominant emotion (Positive or Negative)
    df_emoji_sent["dominant"] = df_emoji_sent.apply(_get_dominant, axis = 1)
    df_emoji_sent["dominant"].value_counts()
    
    # convert to dictionary for fast lookup
    dict_emoji_sent = df_emoji_sent.set_index("demojized")["dominant"].to_dict()
    
    # include emoticon mapping for counting
    dict_emoji_sent.update({e: "Positive" for e in emoticon_map["positive"]})
    dict_emoji_sent.update({e: "Negative" for e in emoticon_map["negative"]})
    
    # emoji-related sentiment extraction
    df[["num_emoji", "num_emoji_pos", "num_emoji_neg"]] = df[text_col].progress_apply(_count_emojis)
    
    print("Emoji and emoticon counts extracted.\n")
    


def process_emojis(df, text_col, emoticon_map = EMOTICON_MAP):
    def _process_emojis(text):
        # normalize unicode (NFKD removes variations)
        text = unicodedata.normalize("NFKD", text)
        # remove variation selectors, causing emoji inconsistencies
        text = text.replace("\ufe0f", " ")
        
        # extract special symbols and emoticons from mapping
        pos_emoticons = set(emoticon_map["positive"] + ["♡"])
        neg_emoticons = set(emoticon_map["negative"])
        pattern = r"(" + r"|".join(map(re.escape, pos_emoticons.union(neg_emoticons))) + r")"
        
        # find all matching emoticons in text
        symb = re.findall(pattern, text)
        
        # exclude copyright symbol (for successive encoding-decoding)        
        dist_emo = [e for e in emoji.distinct_emoji_list(text) if e != "©"]
        list_emo = [e for e in emoji.emoji_list(text) if e["emoji"] != "©"]
        
        # get string of concatenated distinct decoded emojis and emoticons
        emoji_unique = emoji.demojize(" ".join(dist_emo) if text else " ") + " " + " ".join(list(set(symb)))
        
        # concatenate the list of decoded emojis in a string
        emoji_list = emoji.demojize(" ".join([entry["emoji"] for entry in list_emo]) if text else " ") + " " + " ".join(symb)
        
        return pd.Series([emoji_unique, emoji_list])
    
    
    # get decoded unique emojis and list of emojis 
    df[["emoji_unique", "emoji_list"]] = df[text_col].progress_apply(lambda x: _process_emojis(x))
    
    # decode emojis in the text to their names (e.g. :grinning:), except copyright symbol
    df[text_col] = df[text_col].progress_apply(lambda x: emoji.demojize(x).replace(":copyright:", "©"))
    
    print("Emojis processed.\n")



def fix_encoding(df, text_col):    
    def _clean_encoding(text):
        if not isinstance(text, str):
            return ""
        
        try:
            # decode incorrectly encoded text (AtlÃ©tico → Atlético)
            fixed_text = text.encode("latin1", errors = "ignore").decode("utf-8", errors = "ignore")
        except UnicodeEncodeError:
            fixed_text = text
        
        return fixed_text

    df[text_col] = df[text_col].progress_apply(_clean_encoding)

    print("Text encoding fixed.\n")
    
    

def extract_emotions(df, text_col):
    # extract emotion vectors using NRCLex
    def _get_emotions(text):
        emotion = NRCLex(text)
        affect_frequencies_dict = {emotion_class: round(frequency, 2) for emotion_class, frequency in emotion.affect_frequencies.items()}
            
        return pd.Series(affect_frequencies_dict)

    df[["fear", "anger",
        "anticip", "trust",
        "surprise", "positive",
        "negative", "sadness",
        "disgust", "joy",
        "anticipation"]] = df[text_col].progress_apply(_get_emotions)
    
    # drop anticipation feature (not anticip), too many missing values
    df.drop(columns = ["anticipation"], inplace = True)
    
    print("Emotions scores extracted.\n")
    
    

def extract_pol_subj(df, text_col):
    # extract text's polarity and subjectivity
    def _get_pol_subj(text):
        try:
            blob = TextBlob(text)
            pol = blob.sentiment.polarity
            subj = blob.sentiment.subjectivity
        except:
            # neutral values for each
            pol = 0
            subj = 0.5
        
        return pd.Series([pol, subj])
    
    df[["polarity", "subjectivity"]] = df[text_col].progress_apply(_get_pol_subj)
    
    print("Polarity and subjectivity extracted.\n")
    


def extract_VAD(df, text_col, vad = VAD):    
    def _emotion_VAD(text, dim):
        words_VAD = word_tokenize(text)
        score = [vad_dict[i][dim] if i in vad_dict else 0 for i in words_VAD]
        return sum(score) / max(len(score), 1)

    def _analyze_valence(text):
        return _emotion_VAD(text, "valence")

    def _analyze_arousal(text):
        return _emotion_VAD(text, "arousal")

    def _analyze_dominance(text):
        return _emotion_VAD(text, "dominance")
    
    def _get_VAD(text):
        vad = {"valence":_analyze_valence(text),
            "arousal":_analyze_arousal(text),
            "dominance":_analyze_dominance(text)
        }
    
        return pd.Series(vad)
    
    # file VAD translated in python dicts 
    vad.columns = ["word", "valence", "arousal", "dominance"]
    vad_dict = vad.set_index("word").T.to_dict()
    
    df[["valence", "arousal", "dominance"]] = df[text_col].progress_apply(_get_VAD)
    
    print("VAD extracted.\n")
    
    

def extract_readability(df, text_col):
    def _get_readability(text):
        # extract readability scores for 8 tests and number of difficult words
        if not isinstance(text, str) or not text.strip():
            return pd.Series([0] * 9)
        
        return pd.Series([
                        textstat.flesch_reading_ease(text),
                        textstat.flesch_kincaid_grade(text),
                        textstat.gunning_fog(text),
                        textstat.smog_index(text),
                        textstat.automated_readability_index(text),
                        textstat.coleman_liau_index(text),
                        textstat.dale_chall_readability_score(text),
                        textstat.linsear_write_formula(text),
                        textstat.difficult_words(text)
        ])
    
    df[["flesch", "flesch_kincaid", "fog",
       "smog", "ari", "coleman_liau",
       "dale_chall", "linsear", "difficult_words"]] = df[text_col].progress_apply(_get_readability)
    
    print("Readability scores extracted.\n")
    
    

def clean_text(df, text_col):
    # clean the comments before further processing
    def _clean(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        # remove new lines and tabulations
        text = re.sub(r"[\n\t]", " ", text)
        # remove common textual emoji patterns
        emoji_pattern = r"(:-?\)|:-?\(|:-?D|:-?P|;-?\)|<3|XD|:-?/|:-?\|)"
        text = re.sub(emoji_pattern, " ", text)
        
        # remove all punctuation from text
        punct = ['$', '%', '&', '(', ')', '*', '+', '/', '<', '=', '>', '@', '[', '\\',
                 ']', '^', '_', '`', '{', '|', '}', '~', '»', '«', '“', '”', '#', ';', 
                 '!', '?', '.', ',', ':', '"', "'", "-"]
        punct_pattern = re.compile("[" + re.escape("".join(punct)) + "]")
        text = re.sub(punct_pattern, " ", text)
        
        # remove numerical values
        text = re.sub(r"\d+", " ", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        
        return text

    # clean the comments' content column
    df[text_col] = df[text_col].progress_apply(_clean)

    print("Text cleaned.\n")
    


def lowercase(df, text_col):
    df[text_col] = df[text_col].progress_apply(lambda x: x.lower() if isinstance(x, str) else "")
    
    print("Lowercasing done.\n")
    
    

# function to extract count of total words, unique words, adjectives, nouns, verbs, and lexical words (adj, noun or vb)
def extract_word_counts(df, text_col):
    # function to count words based on pos
    def _pos_tagging(text):
        if not isinstance(text, str) or not text.strip():
            return pd.Series([0] * 4)
        
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # number of adjectives
        num_words_adj = sum(1 for _, tag in pos_tags if tag.startswith(("JJ")))
        # number of nouns
        num_words_nouns = sum(1 for _, tag in pos_tags if tag.startswith(("NN")))
        # number of verbs
        num_words_verbs = sum(1 for _, tag in pos_tags if tag.startswith(("VB")))
        # number of full words (adjectives, nouns, verbs)
        num_words_full = num_words_adj + num_words_nouns + num_words_verbs

        return pd.Series([num_words_adj, num_words_nouns, num_words_verbs, num_words_full])
    
    # number of words
    df["num_words"] = df[text_col].progress_apply(lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0)
    # number of unique words in the text
    df["num_words_unique"] = df[text_col].progress_apply(lambda x: len(set(word_tokenize(x.lower()))) if isinstance(x, str) else 0)
    # number of adj
    df[["num_words_adj", "num_words_noun", 
        "num_words_verb", "num_words_lex"]] = df[text_col].progress_apply(_pos_tagging)
    
    print("Word counts retrieved.\n")
    
    

def process_stopwords(df, text_col):
    # get stopwords
    stopw = set(stopwords.words("english"))
    # some negations
    negations = {
        "not", "no", "nor", "n't", "never", "none", "nothing", "nowhere", 
        "neither", "nobody", "without", "cannot", "can't", "won't", "doesn't",
        "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "didn't", "hasn't", 
        "haven't", "hadn't", "mustn't", "shan't", "don't", "ain't",

        # informal variants
        "aint", "gonna not", "wont", "dunno", "nah", "nope",

        # slang & contractions (even if we shouldn't have any left)
        "cant", "dont", "doesnt", "isnt", "wasnt", "shouldnt", 
        "wouldnt", "couldnt", "didnt", "hasnt", "havent", "hadnt", "mustnt", "shant"
        }
    # some pronouns (if you want to keep them, i did not)
    #pron = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'she', 'her', 'it', 'they', 'them'}
    
    # remove negations from stopwords to delete, they are important for sentiment analysis    
    stopw -= negations
    # remove pronouns (if you want to keep them, i did not)
    #stopw -= pron

    # number of stopwords
    df["num_stopw"] = df[text_col].progress_apply(lambda x: sum(1 for w in word_tokenize(x) if w in stopw) if isinstance(x, str) else 0)
    # remove stopwords from text
    df[text_col] = df[text_col].progress_apply(lambda x: " ".join([w for w in word_tokenize(x) if w not in stopw]) if isinstance(x, str) else "")
    
    print("Stopwords counted and removed.\n")
    


def lemmatization(df, text_col):
    # lemmatize text using pos for improved accuracy
    def _lemmatize(text):
        if isinstance(text, str):
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # convert tags for wordnet (n, a, r, v)
            wn_tags = []
            for word, tag in pos_tags:
                if tag.startswith(("JJ")): wn_tags.append((word, "a"))
                elif tag.startswith(("VB")): wn_tags.append((word, "v"))
                elif tag.startswith(("RB")): wn_tags.append((word, "r"))
                else: wn_tags.append((word, "n"))
            
            return " ".join([lemmatizer.lemmatize(word, tag) for word, tag in wn_tags])
        return ""
    
    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    df[text_col] = df[text_col].progress_apply(_lemmatize)

    print("Lemmatization performed.\n")
    
    

def apply_pipeline(df, text_col, pipeline):
    for process in pipeline:
        process(df, text_col)
            
    print("\nPIPELINE APPLIED.\n")
    
    

# ------------------- POST-PROCESSING functions -------------------



def is_latin(text):
    # language detection (keep only probable latin alphabet)
    if not isinstance(text, str) or not text.strip():
        return False  # Skip non-string and empty values
    
    # count latin vs. non-latin characters
    latin_count = sum(1 for char in text if 'LATIN' in unicodedata.name(char, ''))
    total_count = sum(1 for char in text if char.isalpha())  # Count all alphabetic characters
    
    # consider latin text based on threshold on percentage of chars
    return latin_count / total_count >= 0.9 if total_count > 0 else False