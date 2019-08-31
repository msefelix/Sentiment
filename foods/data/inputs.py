import pandas as pd
import re

import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.model_selection import train_test_split

from foods.settings import intermediate, train_formated, dev_formated, source, food_data
from foods.data.augmentation import load_augmentation


def prepare_train_dev(
    dev_size=0.2,
    number_of_augmentation=9000,
    clean_txt=True,
    stem_option=False,
    rem_stop_option=False,
):
    """Prepare training and development data.

    Parameters
    ----------
    dev_size : float, optional
        Ratio of development data, by default 0.2
    number_of_augmentation : int, optional
        Number of augmentation samples to use, by default 9000  
    clean_txt : bool, optional
        Whether to clean the txt, by default True        
    stem_option : bool, optional
        Whether to stem the txt, by default False
    rem_stop_option : bool, optional
        Whether to remove stop words from the txt, by default False

    Returns
    -------
    Transformed datasets
    """

    df = pd.read_table(f"./{source}/{food_data}", header=None).rename(
        columns={0: "text", 1: "label"}
    )[["label", "text"]]

    if clean_txt:
        df = preprocess(df, stem_option=stem_option, rem_stop_option=rem_stop_option)
    df_train, df_dev = train_test_split(
        df, test_size=dev_size, random_state=42, stratify=df["label"]
    )
    df_dev.set_index("label").to_csv(f"./{intermediate}/{dev_formated}")

    if number_of_augmentation > 0:
        df_aug = load_augmentation(number_of_samples=number_of_augmentation)
        if clean_txt:
            df_aug = preprocess(
                df_aug, stem_option=stem_option, rem_stop_option=rem_stop_option
            )
        df_train = pd.concat([df_train, df_aug])

    df_train.set_index("label").to_csv(f"./{intermediate}/{train_formated}")

    return df_train, df_dev, df


def preprocess(df_data, stem_option=False, rem_stop_option=False):
    # Convert to lowercase
    df_data = df_data.apply(lambda x: x.astype(str).str.lower())

    for col in ["text"]:
        # Replace special word
        df_data[col] = df_data[col].apply(lambda x: special_text2word(x))

        # Expand contraction
        df_data[col] = df_data[col].apply(lambda x: decontracted(x))

        # Remove punctuation
        punc_symbol = re.compile(r"[^\w\s]+")
        df_data[col] = df_data[col].apply(lambda x: punc_symbol.sub(" ", x))

        # Wordnet Lemmatizer with appropriate POS tag
        lemmatizer = WordNetLemmatizer()
        df_data[col] = df_data[col].apply(lambda x: lemma(x, lemmatizer))

        # Stem
        if stem_option:
            stemmer = SnowballStemmer("english", ignore_stopwords=True)
            df_data[col] = df_data[col].apply(lambda x: stem(x, stemmer))

        # Remove stopwords
        if rem_stop_option:
            stop_words = stopwords.words("english")
            df_data[col] = df_data[col].apply(lambda x: remstop(x, stop_words))

    return df_data


def special_text2word(phrase):
    for key, value in special_word_dict.items():
        phrase = re.sub(key, value, phrase)
    return phrase


special_word_dict = {
    "the us ": "the america",
    "us dollar": "america dollar",
    "us subprime": "america subprime",
    "us map": "america map",
    "in us": "in america",
    "us culture": "america culture",
    "us treasuries": "america treasuries",
    "us planes": "america planes",
    " usa ": " america ",
    "u.s.": "america",
    "u.s.a.": "america",
    "united states": "america",
    "e-mail": "email",
    "e\.g\.": "eg",
    "\$": " dollar ",
    "\%": " percent ",
    "\&": " and ",
}


def decontracted(phrase):
    # general
    for key, value in CONTRACTION_MAP.items():
        phrase = re.sub(key, value, phrase)

    # special
    phrase = re.sub(r"\'s", "", phrase)

    return phrase


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def lemma(text, lemmatizer):
    text = str(text).split()
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text
    ]
    text = " ".join(lemmatized_words)
    return text


def stem(text, stemmer):
    text = str(text).split()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text


def remstop(text, stop_words):
    text = str(text).split()
    remstop_words = [word for word in text if word not in stop_words]
    text = " ".join(remstop_words)
    return text


CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}
