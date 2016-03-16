# -*- coding: utf-8 -*-
import data
import feature_extraction
from stop_words import get_stop_words
import re
import io
from collections import Counter, OrderedDict
import numpy as np

stop_words_set = set(get_stop_words('en') + get_stop_words('fr') + ['plus'])

print stop_words_set

__author__ = 'David Montoya'

data = data.posts()


def text_preprocessor(text):
    result = [x for x in re.split('\W+', text, flags=re.UNICODE) if x not in stop_words_set and len(x) > 1]
    return result


times = feature_extraction.posts_extract_time(data)
terms = feature_extraction.posts_extract_terms(data, with_stemmer=False, remove_tags=True)
tokenized_terms = map(text_preprocessor, terms)

term_counts = {}

for idx, time in enumerate(times):
    month_year = time.strftime("%Y-%m")
    counter = term_counts.get(month_year, None)
    if counter is None:
        counter = []
        term_counts[month_year] = counter
    counter.append(tokenized_terms[idx])

with io.open('data/words.txt', 'w+', encoding="utf-8") as f:
    f.write(u"<table>")
    for month_year, terms_for_month_year in OrderedDict(sorted(term_counts.items())).iteritems():
        terms_flat = [item for sublist in terms_for_month_year for item in sublist]
        term_count = Counter(terms_flat)

        f.write(u"<tr><td>%s</td><td>%s</td></tr>" % (
        month_year, u", ".join([u"%s: %s" % t for t in term_count.most_common(25)])))
        f.write(u"\n")
    f.write(u"</table>")
