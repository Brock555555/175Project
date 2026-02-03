import create_dataset

if __name__ == '__main__':
    urban_dictionary = create_dataset.load_urban_dataset()
    print(urban_dictionary.shape)