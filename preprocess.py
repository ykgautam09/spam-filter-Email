import numpy as np
import pandas as pd
import pickle
from configparser import ConfigParser
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.ensemble import RandomForestClassifier

config = ConfigParser()
config.read('./config.ini')
DATA_PATH = config['DEFAULT']['DATA_PATH']
VECTORIZER_PATH = config['DEFAULT']['VECTORIZER_PATH']
MODEL_PATH = config['DEFAULT']['MODEL_PATH']


def process_data(data):
    phone_regex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
    date_regex = re.compile(r'\d\d\\d\d\\d\d\d\d')
    time_regex = re.compile(r'\d\d:\d\d')
    processed_mail = []
    lemmetizer = WordNetLemmatizer()
    for sentence in data:
        l = []
        for i in phone_regex.findall(sentence):
            sentence = sentence.replace(i, 'telphonetel')
        for i in date_regex.findall(sentence):
            sentence = sentence.replace(i, 'teldatetel')
        for i in time_regex.findall(sentence):
            sentence = sentence.replace(i, 'teltimetel')
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        li = []
        for word in sentence.split(' '):
            if word not in stopwords.words('english'):
                word = lemmetizer.lemmatize(word)
                li.append(word)
        li = list(filter(lambda x: x != '', li))
        sentence = ' '.join(li)
        processed_mail.append(sentence)
        print(len(processed_mail), sentence)
    return np.array(processed_mail)


def generate_model(x, y, model_path):
    model = RandomForestClassifier()
    rf = model.fit(x, y)
    with open(model_path, 'wb')as f:
        pickle.dump(rf, f, protocol=2)
        print('model generated')


def preprocess_data(data_path, vectorizer_path, model_path):
    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.sample(frac=1).reset_index(drop=True)  # data shuffle

    processed_data = process_data(raw_data['text'])

    tokenizer = Tokenizer(oov_token='_OOV_')
    tokenizer.fit_on_texts(processed_data)
    text_sequence = tokenizer.texts_to_sequences(processed_data)
    x_padded_data = pad_sequences(text_sequence, maxlen=4000, padding='post', truncating='post')

    with open(vectorizer_path, 'wb')as f:
        pickle.dump(tokenizer, f, protocol=2)
        print('vectorizer generated')
    generate_model(x_padded_data, raw_data['spam'], model_path)


if __name__ == '__main__':
    preprocess_data(DATA_PATH, VECTORIZER_PATH, MODEL_PATH)
    # print(get_stat())
