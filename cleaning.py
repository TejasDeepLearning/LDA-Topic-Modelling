import pandas as pd 
import numpy as np 
import re
import pickle

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

import gensim

# List of stopwords in English
stop = stopwords.words('english')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initilalize the stemmer
stemmer = PorterStemmer()

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(
    r"C:\Users\tejas\Documents\Datasets\comcast_consumeraffairs_complaints\comcast_consumeraffairs_complaints.csv"
)

train_data = df.sample(frac=0.9)
test_data = df.drop(train_data.index)

test_data.to_csv(r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\test_data.csv",
                 index=False)

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

texts = train_data['text'].apply(clean)

train_data = pd.concat([texts, train_data.rating], axis=1)

docs = train_data['text'].tolist()

with open(r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\doc_list", "wb") as f:
    pickle.dump(docs, f)