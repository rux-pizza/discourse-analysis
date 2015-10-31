__author__ = 'BH4101'
# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from bs4 import BeautifulSoup, NavigableString, Tag
import numpy as np
from nltk.stem import snowball
from sklearn.cross_validation import train_test_split

vectorizer = TfidfVectorizer(analyzer="word", strip_accents='unicode',
                             tokenizer=None, encoding=u'utf-8',
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)
with_stemmer = True
french_stemmer = snowball.FrenchStemmer()
english_stemmer = snowball.EnglishStemmer()
import datetime
data = pd.read_csv("data/posts.csv", dtype={'cooked': np.str}, na_values=[], keep_default_na=False, encoding="utf-8")


def html_to_text(html):
    if not len(html.contents):
        return
    stop_node = html._last_descendant().next_element
    node = html.contents[0]
    while node is not stop_node:
        if isinstance(node, NavigableString):
            yield node.string
        elif isinstance(node, Tag):
            if node.name == "a":
                yield " htmllink "
            elif node.name == "img":
                yield " htmlimg "
            elif node.name == "br":
                yield "\n"
        node = node.next_element


def preprocess_text(raw_text):
    html = BeautifulSoup(raw_text, "html.parser")
    stripped_text = u"".join([s for s in html_to_text(html)])
    no_httpaddr = re.sub(r"\b(http|https)://[^\s]*", "httpaddr", stripped_text, flags=re.UNICODE)
    no_numbers = re.sub(r"\b[0-9]+\b", "parsednumber", no_httpaddr, flags=re.UNICODE)
    letters_only = re.sub(r"(\W|_)", " ", no_numbers, flags=re.UNICODE)
    words = letters_only.lower().split()
    meaningful_words = words
    if with_stemmer:
        meaningful_words = map(french_stemmer.stem, meaningful_words)
        meaningful_words = map(english_stemmer.stem, meaningful_words)

    return " ".join(meaningful_words)

text_vector = []
time_data = []

print "Importing data"
for i in xrange(0, data["cooked"].size):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    d = preprocess_text(data["cooked"][i])
    text_vector.append(d)
    post_time = data["post_time"][i]
    post_time = datetime.datetime.strptime(re.sub(r"\.[0-9]+$","",post_time), "%Y-%m-%d %H:%M:%S")
    seconds_since_midnight = (post_time - post_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()/(60*60*24)
    time_data.append(seconds_since_midnight)

label_data = data["users_giving_a_fuck"] != ""
user_data = data["posting_user"]

text_vector_train, text_vector_test, label_train, label_test, user_train, user_test, time_train, time_test = train_test_split(text_vector, label_data, user_data, time_data, train_size=0.8)

print "Training feature extractor"
le = preprocessing.LabelEncoder()
le.fit(user_data)
from scipy.sparse import csr_matrix, vstack, hstack
features_train = hstack((np.transpose(csr_matrix(le.transform(user_train))), np.transpose(csr_matrix(time_train)), vectorizer.fit_transform(text_vector_train)))
print vectorizer.get_feature_names()
features_test = hstack((np.transpose(csr_matrix(le.transform(user_test))), np.transpose(csr_matrix(time_test)), vectorizer.transform(text_vector_test)))

scaler = preprocessing.StandardScaler(with_mean=False).fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

print "Training the model"
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC

classifiers = [LinearSVC(C=0.01),
               LinearSVC(C=0.1),
               LinearSVC(C=0.3),
               LinearSVC(C=1),
               LinearSVC(C=3),
               LinearSVC(C=10),
               LinearSVC(C=15),
               LinearSVC(C=30),
               LinearSVC(C=50),
               LinearSVC(C=80),
               LinearSVC(C=100),
               LinearSVC(C=300),
               LinearSVC(C=1000),
               DummyClassifier(strategy="most_frequent"),
               DummyClassifier(strategy="stratified"),
               SVC(gamma=2,C=1),
               SVC(gamma=2,C=10)]


def fit(classifier):
    print classifier
    classifier.fit(features_train, label_train)
    predicted = classifier.predict(features_test)
    print "accuracy_score: %s" % accuracy_score(label_test, predicted)
    print "f1_score: %s" % f1_score(label_test, predicted)
    print "confusion_matrix: %s" % confusion_matrix(label_test, predicted)

map(fit, classifiers)
