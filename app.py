import pandas as pd 
import numpy as np 
import re
import pickle
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

import gensim

import streamlit as st

# List of stopwords in English
stop = stopwords.words('english')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initilalize the stemmer
stemmer = PorterStemmer()

def clean(text):
    # Convert to string
    text = str(text)
    # Convert text to lowercase
    text = text.lower() 
    # Remove mentions, URLs, special characters, and hashtags using regular expressions
    text = re.sub(
        r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|#[A-Za-z0-9_]+", "", text
    )
    # Remove stopwords from the text
    text = " ".join([word for word in text.split() if word not in (stop)])
    # Lemmatize the text to reduce words to their base form, 
    # then apply stemming to reducem words to their root form
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > 3:
            result.append(stemmer.stem(lemmatizer.lemmatize(token)))
            
    return result

with open(
    r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\doc_list", "rb"
) as f:
    docs = pickle.load(f)
    
lda = gensim.models.ldamodel.LdaModel.load(
    r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\lda_model"
)

    
dictionary = gensim.corpora.Dictionary(docs)

dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

unseen_document = st.text_area(label='Enter text here', 
                               value='')

dictionary = gensim.corpora.Dictionary(docs)

dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(clean(unseen_document))

name = [
    'Contract and Pricing',
    'Customer Experience',
    'Payments and Billing',
    'Installation and Service Appointments',
    'Internet Speed and Connectivity'
]

topics = lda[bow_vector]
for id, prob in topics:
    if id == 0:
        top = name[0]
    elif id == 1:
        top = name[1]
    elif id == 2:
        top = name[2]
    elif id == 3:
        top = name[3]
    else:
        top = name[4]
    st.write(f"Topic: {top}")
    st.write(f"Probability: {prob}")