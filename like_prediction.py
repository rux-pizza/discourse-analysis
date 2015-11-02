__author__ = 'BH4101'
import data
import feature_extraction

data = data.posts()

features_train, features_test, label_train, label_test = feature_extraction.extract_features(data, train_size=0.8, with_stemmer=True, tfidf=True)


print "Training the model"
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC


baseline_classifiers = [DummyClassifier(strategy="most_frequent"), DummyClassifier(strategy="stratified")]
C_range = [0.01, 0.1, 0.3, 1, 3, 10, 15, 30, 50, 80, 100, 300, 1000]


def linear_svg(C):
    return LinearSVC(C=C)

classifiers = baseline_classifiers
classifiers.extend(map(linear_svg, C_range))


def fit(classifier):
    classifier.fit(features_train, label_train)
    predicted = classifier.predict(features_test)
    accuracy = accuracy_score(label_test, predicted)
    f1 = f1_score(label_test, predicted)
    print "%s, accuracy=%s, f1_score=%s" % (classifier, accuracy, f1)


for classifier in classifiers:
    fit(classifier)
