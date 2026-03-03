import os
import numpy as np
import pandas as pd
from datasets import Dataset

DATA_DIR = os.path.join(os.pardir, '175Project')

COL_NAMES = ['character', 'browsing_page_url', 'word_url', 'word', 'definition', 'sentence']

def load_urban_dataset():
    file_paths = []
    for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'Urban')):
        for f in files:
            if f.endswith('.csv') and f.startswith('urban_data'):
                file_paths.append(os.path.join(root, f))
    df_urban = pd.concat([pd.read_csv(f, names=COL_NAMES) for f in file_paths])

    df_nulls = df_urban[(df_urban.isnull().any(axis=1)) | (df_urban.isna().any(axis=1))]
    df_urban = df_urban.drop(df_nulls.index)

    return df_urban

def prepare_dataset(df, min_definition=5, max_definition=300, min_word=2, max_word=150):
    # enforce definition sizes
    df = df[df["definition"].str.len() > min_definition]
    df = df[df["definition"].str.len() < max_definition]

    # enforce word sizes
    df = df[df["word"].str.len() > min_word]
    df = df[df["word"].str.len() < max_word]

    texts = [f"[DEF] {definition} [IDM] {word}" for definition, word in zip(df["definition"], df["word"])]

    return Dataset.from_dict({"text": texts})
