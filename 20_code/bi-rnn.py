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
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#initialize review parameters
max_words = 5000
max_len = 200

def plot_epochs(hist, name):
    '''
    Plotting function that visualizes the trend in accuracy over epochs
    '''
    fig = plt.plot(hist.history['accuracy'])
    title = plt.title(f"Accuracy of Epochs history ({name} data)")
    xlabel = plt.xlabel("Epochs")
    ylabel = plt.ylabel("Accuracy")
    return fig

def run_disc(dataset, data_name = 'real'):
    '''
    Function tokenizes data, splits into train/test data, build, fits and models a bi-directional recurrent neural network
    Input: all data used in train and test (DataFrame)
    Output: Accuracy of model on testing data (string)
    '''
    #if synthetic data, rename column to match real data
    if data_name != 'real':
        dataset.rename(columns={'text':'clean_text'}, inplace=True)

    #seperate X and y
    all_text = dataset['clean_text']
    Y = dataset['label']

    #initialize tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(all_text)
    sequences = tokenizer.texts_to_sequences(all_text)
    reviews = pad_sequences(sequences, maxlen=max_len)
    vocab_size = len(tokenizer.word_index) + 1

    #80/20 split of train/test data
    X_train, X_test, Y_train, Y_test = train_test_split(reviews, Y, test_size = 0.2, random_state = 42)

    #build bi-directional LSTM
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
    #print(model.summary())

    #fit model on training data
    history = model.fit(X_train, Y_train.values, epochs=10, verbose=1, batch_size=64)

    #plot accuracy
    f = plot_epochs(history, data_name)
    plt.show()

    #predict and evaluate results on testing data
    scores = model.evaluate(X_test, Y_test, verbose=0)
    output_acc = ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return output_acc

if __name__ == '__main__':
    all_data = pd.read_csv('../10_cleaned_data/processed_text.csv')
    print("Accuracy of Model fit on real data ", run_disc(all_data))
    synth_data = pd.read_csv('../00_source_data/synthetic_data_50k.csv')
    print("Accuracy of Model fit on synthetic data ", run_disc(synth_data, data_name = 'synthetic'))
