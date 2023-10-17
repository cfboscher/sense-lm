from spellchecker import SpellChecker
from tqdm import tqdm
import spacy


import pandas as pd
import re
import string

stopwords= {'im', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
            'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'own', 'same',
            'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
            'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 'didn', 'doesn', 'hadn', "hadn't",
            'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
            'needn', "needn't", 'shan', "shan't", 'shouldn', 'wasn', "wasn't", 'weren', "weren't", 'won', 'wouldn'}

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
#     text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join(e.lower() for e in text.split() if e.lower() not in stopwords)
    return text

def spellcheck(sentence):
    spell = SpellChecker()
    sentence = sentence.split()

    corrected_sentence = [spell.correction(word) if spell.correction(word) is not None else word for word in sentence]
    return ' '.join(corrected_sentence) if corrected_sentence is not None else sentence


def normalize(text, spacy_model, lowercase=True):
    if lowercase:
        text = text.lower()
    comment = spacy_model(text)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not word.is_stop and not word.is_punct and word.pos_ in ['NOUN', 'ADJ', 'ADV', 'VERB']:
                lemmatized.append(lemma)
    return " ".join(lemmatized)


def preprocess_text(df):
    tqdm.pandas()

    spacy_model = spacy.load("en_core_web_md")

    df['spellchecked_text'] = df['text'].progress_apply(spellcheck)
    df['normalized_text'] = df['spellchecked_text'].progress_apply(normalize, spacy_model=spacy_model)
    df['normalized_text'] = df['normalized_text'].apply(lambda x: clean_text(x))
    return df

def preprocess_column(df, column, dest):
    tqdm.pandas()

    spacy_model = spacy.load("en_core_web_md")

    df['spellchecked_text'] = df[column].progress_apply(spellcheck)
    df[dest] = df['spellchecked_text'].progress_apply(normalize, spacy_model=spacy_model)
    df[dest] = df[dest].apply(lambda x: clean_text(x))
    return df