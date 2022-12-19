#import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout


#initialize review parameters
max_words = 5000
max_len = 200

def run_disc(dataset):

    all_text = dataset['clean_text']
    Y = data['label']

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(all_text)
    sequences = tokenizer.texts_to_sequences(all_text)
    reviews = pad_sequences(sequences, maxlen=max_len)

    X_train, X_test, Y_train, Y_test = train_test_split(reviews, Y, test_size = 0.2, random_state = 42)

    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_len))
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(None, 1))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(learning_rate=0.01)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        optimizer=adam, 
        metrics=['accuracy']
    )
    print(model.summary())

    history = model.fit(X_train, Y_train.values, epochs=10, verbose=1, batch_size=64)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    output_acc = ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return output_acc

if __name__ == '__main__':
    print("Model fit on real data results")
    all_data = pd.read_csv('../10_cleaned_data/processed_text.csv')
    run_disc(all_data)
    print("Model run on synthetic data")
    synth_data = pd.read_csv('../10_cleaned_data/processed_text.csv')
    run_disc(all_data)