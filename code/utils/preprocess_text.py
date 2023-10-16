from spellchecker import SpellChecker
from tqdm import tqdm
import spacy


import pandas as pd
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
    return df
