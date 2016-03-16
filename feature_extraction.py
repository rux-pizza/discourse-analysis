# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup, NavigableString, Tag
from nltk.stem import snowball
import re
import datetime
import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split

__author__ = 'David Montoya'

stemmers = [snowball.FrenchStemmer(), snowball.EnglishStemmer()]


def vectorizer(tfidf, strip_accents='unicode', stop_words=None):
    if tfidf:
        return TfidfVectorizer(analyzer="word", strip_accents=strip_accents,
                               tokenizer=None, encoding=u'utf-8',
                               preprocessor=None,
                               stop_words=stop_words,
                               max_features=5000)
    else:
        return CountVectorizer(analyzer="word", strip_accents=strip_accents,
                               tokenizer=None, encoding=u'utf-8',
                               preprocessor=None,
                               stop_words=stop_words,
                               max_features=5000)


def posts_extract_terms(data, with_stemmer, remove_tags=False):
    def text_preprocessor(text):
        return preprocess_text(text, with_stemmer=with_stemmer, remove_tags=remove_tags)

    text_vector = map(text_preprocessor, data["cooked"])

    return text_vector


def posts_extract_time(data):
    def post_time_extractor(post_time):
        post_timestamp = datetime.datetime.strptime(re.sub(r"\.[0-9]+$", "", post_time), "%Y-%m-%d %H:%M:%S")
        return post_timestamp

    return map(post_time_extractor, data["post_time"])


def posts_extract_time_since_midnight(data):
    def post_second_since_midnight_extractor(post_time):
        post_timestamp = datetime.datetime.strptime(re.sub(r"\.[0-9]+$", "", post_time), "%Y-%m-%d %H:%M:%S")
        seconds_since_midnight = (post_timestamp - post_timestamp.replace(hour=0, minute=0, second=0, microsecond=0))
        seconds_since_midnight = seconds_since_midnight.total_seconds()/(60*60*24)
        return seconds_since_midnight

    return map(post_second_since_midnight_extractor, data["post_time"])


def extract_features(data, train_size, with_stemmer, tfidf):
    print "Vectorizing posts..."
    posting_user_vector = data["posting_user"]
    post_time_vector = posts_extract_time_since_midnight(data)
    text_vector = posts_extract_terms(data, with_stemmer=with_stemmer)
    label_vector = extract_labels(data)

    text_vector_train, text_vector_test, posting_user_vector_train, posting_user_vector_test, post_time_vector_train, post_time_vector_test, label_vector_train, label_vector_test = \
        train_test_split(text_vector, posting_user_vector, post_time_vector, label_vector, train_size=train_size)

    print "Extracting features..."
    vec = vectorizer(tfidf)
    le = preprocessing.LabelEncoder()
    le.fit(posting_user_vector)
    from scipy.sparse import csr_matrix, hstack

    def prepare_matrix(vector):
        return np.transpose(csr_matrix(vector))
    features_vector_train = hstack((prepare_matrix(le.transform(posting_user_vector_train)),
                                    prepare_matrix(post_time_vector_train),
                                    vec.fit_transform(text_vector_train)))
    print vec.get_feature_names()
    features_vector_test = hstack((prepare_matrix(le.transform(posting_user_vector_test)),
                                   prepare_matrix(post_time_vector_test),
                                   vec.transform(text_vector_test)))

    scaler = preprocessing.StandardScaler(with_mean=False).fit(features_vector_train)
    features_vector_train = scaler.transform(features_vector_train)
    features_vector_test = scaler.transform(features_vector_test)
    return features_vector_train, features_vector_test, label_vector_train, label_vector_test


def extract_labels(data):
    return data["users_liking"] != ""

tags_to_keep = {"a", "img", "strong", "blockquote", "i"}


def html_to_text(html, remove_tags=False):
    if not len(html.contents):
        return
    stop_node = html._last_descendant().next_element
    node = html.contents[0]
    while node is not stop_node:
        if isinstance(node, NavigableString):
            yield node.string
        elif isinstance(node, Tag):
            if node.name == "br":
                yield "\n"
            elif (not remove_tags) and (node.name in tags_to_keep):
                yield " htmltag%s " % node.name
            elif (not remove_tags) and node.name == "span":
                clazz = node.get("class", "")
                if len(clazz) == 2 and clazz[0] == "typefaces-tag":
                    yield " typefacestag%s " % clazz[1]
            elif node.name == "div":
                clazz = node.get("class", "")
                if "lightbox-wrapper" in clazz:
                    if not remove_tags:
                        yield " htmltagimg "
                    node = node.next_sibling
                    continue
            elif node.name == "aside":
                clazz = node.get("class", "")
                if "onebox" in clazz:
                    if not remove_tags:
                        yield " htmlonebox "
                    node = node.next_sibling
                    continue
        node = node.next_element


def preprocess_text(raw_text, with_stemmer, remove_tags):
    html = BeautifulSoup(raw_text, "html.parser")
    if remove_tags:
        stripped_text = u"".join([s for s in html_to_text(html, remove_tags=True)])
    else:
        stripped_text = u"".join([s for s in html_to_text(html)])
    no_emailaddr = re.sub(r"\b[^\s]+@[^\s]*\.[^.\s]+\b", "" if remove_tags else "emailaddr", stripped_text,
                          flags=re.UNICODE)
    no_httpaddr = re.sub(r"\b((http|https):)?//[^\s]+", "" if remove_tags else "httpaddr", no_emailaddr,
                         flags=re.UNICODE)
    no_numbers = re.sub(r"\b[0-9]+\b", "" if remove_tags else "parsednumber", no_httpaddr, flags=re.UNICODE)
    letters_only = re.sub(r"(\W|_)", " ", no_numbers, flags=re.UNICODE)
    words = letters_only.lower().split()
    meaningful_words = words
    if with_stemmer:
        for stemmer in stemmers:
            def stem(word):
                if not remove_tags and (word.startswith("htmltag") or word.startswith("typefacestag")):
                    return word
                else:
                    return stemmer.stem(word)
            meaningful_words = map(stem, meaningful_words)

    return " ".join(meaningful_words)