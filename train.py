import pandas as pd 
import numpy as np 
import re
import pickle

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

import gensim
from gensim.models import LdaModel

with open(
    r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\doc_list", "rb"
) as f:
    docs = pickle.load(f)
    
dictionary = gensim.corpora.Dictionary(docs)
    
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

document_num = 20
bow_doc_x = bow_corpus[document_num]

    
lda_model =  LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    passes=20,
    num_topics=5,
    iterations=400,
    eval_every=None,
    eta='auto',
    alpha='auto',
    chunksize=2000
)

lda_model.save(r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\lda_model")