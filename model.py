import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df_list = []

df_twitter = pd.read_csv('training.csv', names=['target', 't_id', 'created_at', 'query', 'user', 'text'], sep=',', encoding='utf8')

sentences = df_twitter['text'].values
y = df_twitter['target'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

sentences = df_twitter['text'].values
y = df_twitter['target'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.10, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
classifier = LogisticRegression(max_iter=1600000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Accuracy for twitter dataset: {:.4f}'.format(score))
pickle.dump(classifier, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))