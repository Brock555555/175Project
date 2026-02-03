import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.pardir, '175Project')

COL_NAMES = ['character', 'browsing_page_url', 'word_url', 'word', 'definition', 'sentence']

def load_urban_dataset():
    file_paths = []
    for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'Urban')):
        for f in files:
            if f.endswith('.csv') and f.startswith('urban_data'):
                file_paths.append(os.path.join(root, f))
    df_urban = pd.concat([pd.read_csv(f, names=COL_NAMES) for f in file_paths])
    return df_urban