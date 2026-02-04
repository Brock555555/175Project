import create_dataset
import re

if __name__ == '__main__':
    urban_dictionary = create_dataset.load_urban_dataset()
    print(f"Shape of urban dictionary dataset: {urban_dictionary.shape}")
    ud_sample = urban_dictionary[['word', 'definition', 'sentence']].sample(1)
    for i in ud_sample.values:
        print("Word: ", i[0])
        print("Meaning: ", i[1])
        print("Sentence: ", i[2])