import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv("/workspaces/nlp_final/10_cleaned_data/processed_text.csv")


tfidf = TfidfVectorizer()
MNB = MultinomialNB()
text_count_2 = tfidf.fit_transform(dataset['text'])
vocabulary = tfidf.get_feature_names_out()
x_train, x_test, y_train, y_test = train_test_split(text_count_2, dataset['label'], test_size = 0.25, random_state=5)
MNB.fit(x_train, y_train)



#setting number of entries in our synthetic data
synth_data_len = 50000

#setting the probabilities to use when creating synthetic reviews
prior_prob = np.exp(MNB.feature_log_prob_)

#0 is negative, 1 is positive
labels = [0, 1]


#randomly appending words for synthetic reviews based on probabilities from real-data MNB
df = None
for lab_val in labels:
    synth_sent = []
    for n in range(synth_data_len // 2):
        rand_sentence = random.choices(vocabulary, prior_prob[lab_val], k = random.randint(5, 500))
        synth_sent.append(" ".join(rand_sentence))
    if df is None:
        df = pd.DataFrame({"text": synth_sent, "label":lab_val})
    else: 
        df_other = pd.DataFrame({"text": synth_sent, "label":lab_val})
synth_data = pd.concat([df, df_other])


'''
saving data: if you want to see example synthetic data, please change line 23 to say 
"synth_data_len = 10", otherwise it will take about five minutes to create 50,000 synthetic reviews 

'''
synth_data.to_csv('/workspaces/nlp_final/00_source_data/synthetic_data.csv')