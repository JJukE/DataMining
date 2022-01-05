# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:14:44 2021

@author: jh980
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from textblob import TextBlob
from nltk import tokenize

JNJ=pd.read_csv('C:/Users/jh980/Downloads/JNJ_data.csv')
JNJ=JNJ.astype(str)

content=JNJ['content']
title=JNJ['title']

JNJ['content_polarity'] = JNJ['content'].apply(lambda content: TextBlob(content).sentiment.polarity)

JNJ['title_polarity'] = JNJ['title'].apply(lambda title: TextBlob(title).sentiment.polarity)



JNJ.to_csv('JNJ-textblob.csv', index=False)
