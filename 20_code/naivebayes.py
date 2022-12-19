import pandas as pd



def build_whole_model(dataset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from sklearn import metrics
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.tokenize import RegexpTokenizer


    tfidf = TfidfVectorizer()
    MNB = MultinomialNB()

    text_count_2 = tfidf.fit_transform(dataset['text'])

    x_train, x_test, y_train, y_test = train_test_split(text_count_2, dataset['label'], test_size = 0.25, random_state=5)
    MNB.fit(x_train, y_train)
    accuracy_score_mnb = metrics.accuracy_score(MNB.predict(x_test), y_test)
    print('accuracy score with tf-idf multinomial: ' +str('{:4.2f}'.format(accuracy_score_mnb*100)) + '%')


build_whole_model(pd.read_csv("/workspaces/nlp_final/10_cleaned_data/processed_text.csv"))
build_whole_model(pd.read_csv("/workspaces/nlp_final/00_source_data/synthetic_data_50k.csv"))