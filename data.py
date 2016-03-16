# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

__author__ = 'David Montoya'


def posts():
    print "Reading posts..."
    data = pd.read_csv("data/posts.csv", dtype={'cooked': np.str}, na_values=[],
                       keep_default_na=False, encoding="utf-8")
    return data