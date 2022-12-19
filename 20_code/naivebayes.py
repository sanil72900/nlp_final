import pandas as pd



def build_whole_model(dataset):
    #We import time and begin a timer to understand how long the function takes to run
    import time
    start_time = time.time()
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from sklearn import metrics
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split


    #TfidVectorizer() uses Tf-idf to tokenize the reviews
    tfidf = TfidfVectorizer()

    #This initializes the multinomial naive bayes classification model.
    MNB = MultinomialNB()

    #Here we're transforming our reviews by tokenizing using Tf-idf
    text_count_2 = tfidf.fit_transform(dataset['text'])

    #This splits our data into test and train with 25% used for testing and 75% used for training
    x_train, x_test, y_train, y_test = train_test_split(text_count_2, dataset['label'], test_size = 0.25, random_state=5)
    
    #This line fits the model: by default we are using laplace smoothing and no prior probabilities
    MNB.fit(x_train, y_train)

    #Here we compute accuracy and print it as well as the time it took to run the function
    accuracy_score_mnb = metrics.accuracy_score(MNB.predict(x_test), y_test)
    print('accuracy score with tf-idf multinomial: ' +str('{:4.2f}'.format(accuracy_score_mnb*100)) + '%')
    print("My program took", time.time() - start_time, "to run")


'''
When running this file, the first output is the model that was trained with the real data and the
second output is the model that was trained with the synthetic data.
'''

build_whole_model(pd.read_csv("/workspaces/nlp_final/10_cleaned_data/processed_text.csv"))
build_whole_model(pd.read_csv("/workspaces/nlp_final/00_source_data/synthetic_data_50k.csv"))


'''

code source: https://scikit-learn.org/stable/modules/naive_bayes.html
and
https://www.datacamp.com/tutorial/naive-bayes-scikit-learn

'''

def build_plots(dataset): 
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from sklearn import metrics
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split

    tfidf = TfidfVectorizer()
    MNB = MultinomialNB()
    text_count_2 = tfidf.fit_transform(dataset['text'])
    x_train, x_test, y_train, y_test = train_test_split(text_count_2, dataset['label'], test_size = 0.25, random_state=5)
    MNB.fit(x_train, y_train)
    
    #define metrics
    import matplotlib.pyplot as plt
    #define metrics
    y_pred_proba = MNB.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.title('ROC with ___ Data')
    plt.savefig('/workspaces/nlp_final/30_for_plots/plot.png')
    return 1

#build_plots(pd.read_csv("/workspaces/nlp_final/00_source_data/synthetic_data_50k.csv"))
#build_plots(pd.read_csv("/workspaces/nlp_final/10_cleaned_data/processed_text.csv"))
'''
To build plots, uncomment each build_plots line one at a time. The .png for the model's plot is stored in the 
../30_for_plots folder.


source for plotting code: https://stackoverflow.com/questions/34564830/roc-curve-with-sklearn-python
'''