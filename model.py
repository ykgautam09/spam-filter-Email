import pickle
from configparser import ConfigParser
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import process_data

config = ConfigParser()
config.read('./config.ini')
VECTORIZER_PATH = config['DEFAULT']['VECTORIZER_PATH']
MODEL_PATH = config['DEFAULT']['MODEL_PATH']
DEFAULT_MAIL = config['DEFAULT']['DEFAULT_EMAIL']

with open(VECTORIZER_PATH, 'rb')as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, 'rb')as f:
    model = pickle.load(f)


def predict_mail(email=DEFAULT_MAIL, vectorizer=vectorizer, model=model):
    processed_data = process_data([email])
    text_sequence = vectorizer.texts_to_sequences(processed_data)
    email_vector = pad_sequences(text_sequence, maxlen=4000, padding='post', truncating='post')
    output = model.predict(email_vector)
    return output


if __name__ == '__main__':
    print(predict_mail())
