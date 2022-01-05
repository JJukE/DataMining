# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:14:00 2021

@author: jh980
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

JNJ=pd.read_csv('C:/Users/jh980/Downloads/JNJ_data.csv')
JNJ=JNJ.astype(str)

vader = SentimentIntensityAnalyzer()
title=JNJ['title']
content=JNJ['content']


JNJ['title_scores'] = JNJ['title'].apply(lambda title: vader.polarity_scores(title))
JNJ['title_compound'] = JNJ['title_scores'].apply(lambda title_score_dict: title_score_dict['compound'])


JNJ['content_scores'] = JNJ['content'].apply(lambda content: vader.polarity_scores(content))
JNJ['content_compound'] = JNJ['content_scores'].apply(lambda content_score_dict: content_score_dict['compound'])


JNJ.to_csv('JNJ-vader.csv', index=False)
