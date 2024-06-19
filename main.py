import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bag_of_words import run_bag_of_words
from tf_idf import run_tf_idf
from t_sne import run_tsne
from preprocessing import run_preprocessing
from clustering import run_kmeans
from topic_modeling import run_topic_modeling
from word2vec import run_word2vec
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

default_file_path = './data/text.csv'
# emotion_map = {
#     0: 'sadness',
#     1: 'joy',
#     2: 'love',
#     3: 'anger',
#     4: 'fear',
#     5: 'surprise'
# }

# Load file
if os.path.exists(default_file_path):
    data = pd.read_csv(default_file_path)
else:
    updata_file_path =  input("File not found. Provide new path: ")

    if os.path.exists(updata_file_path):
        data = pd.read_csv(updata_file_path)
    
    else:
        print("Invaild path")
        exit()

data = data.drop(data.columns[0], axis=1)
# data.label = data.label.map(emotion_map)

# Choose which techniques to run
print("Choose the techniques to run:")
print("1. Bag of Words")
print("2. TF-IDF")
print("3. Word2Vec")
print("4. Topic Modeling")
print("5. Clustering")
print("6. T-SNE")
techniques = input("Enter number: ")

if techniques not in {'1', '2', '3', '4', '5', '6'}:
    print("Invalid option. Choose 1, 2, 3, 4, 5 or 6")
    exit()

# Prompt user to choose sampling size
print("Choose the sampling method:")
print("1. Custom sample size")
print("2. Use 0.7/0.3 split")
print("3, Custom sample size for Topic Modelling/Clustering/T-SNE")
print("4. Origin data size for Topic Modelling/Clustering/T-SNE")
split = input("Enter number: ")

if split == "1": # sample size
    sample_size = int(input("Sample size: "))

    if sample_size < len(data):
        data_sample = data.sample(n=sample_size, random_state=42)
        train_data = data_sample.sample(frac=0.8, random_state=42)
        test_data = data_sample.drop(train_data.index)
    else:
        print("Sample size bigger than data size")
        exit()

elif split == "2": # 0.7 / 0.3 split
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

elif split == "3": # sample size for Topic Modelling/Clustering/T-SNE
    sample_size = int(input("Sample size: "))

    if sample_size < len(data):
        data_sample = data.sample(n=sample_size, random_state=42)
        data_text = data_sample.text
        data_label = data_sample.label
    else:
        print("Sample size bigger than data size")
        exit()

elif split == "4": # Origin data size Topic Modelling/Clustering/T-SNE
    data_text = data.text
    data_label = data.label

else:
    print("Invalid option. Choose 1, 2, 3 or 4")
    exit()
print("Finish train and test data split!\n")

if techniques not in {"4", "5", "6"}:
    # Extraction of texts and sentiments
    train_text = train_data.text
    train_label = train_data.label
    test_text = test_data.text
    test_label = test_data.label

    # preprocessing
    train_text = train_text.apply(run_preprocessing)
    test_text = test_text.apply(run_preprocessing)
    print("Finish data preprocessing!\n")

if techniques == '1': # Bag of Words
    print("Choose the ML model: LogR, DT, XGB, LGB, MNB or all")
    models = input("Enter the model: ").split(', ')
    # 对比一下 test_text和test_text.tolist()哪个好
    run_bag_of_words(train_text, train_label, test_text, test_label, models)

elif techniques == '2': # TF-IDF
    print("Choose the ML model: LogR, DT, XGB, LGB, MNB or all")
    models = input("Enter the model: ").split(', ')

    run_tf_idf(train_text, train_label, test_text, test_label, models)

elif techniques == '3': # Word2Vec
    print("Choose the ML model: LogR, DT, XGB, LGB, MNB or all")
    models = input("Enter the model: ").split(', ')

    run_word2vec(train_text, train_label, test_text, test_label, models)

elif techniques == '4': # Topic Modeling
    run_topic_modeling(data_text)

elif techniques == '5': # Clustering in tfidf
    print("Choose the wordembedding techniques to process:")
    print("1. Bag of Words")
    print("2. TF-IDF")

    wordembedding = input("Enter number: ")
    run_kmeans(data_text, wordembedding)

elif techniques == '6': # T-SNE
    print("Choose the wordembedding techniques to process:")
    print("1. Bag of Words")
    print("2. TF-IDF")

    wordembedding = input("Enter number: ")
    run_tsne(data_text, data_label, wordembedding)

else:
    print("Choose 1, 2, 3, 4, 5 or 6")