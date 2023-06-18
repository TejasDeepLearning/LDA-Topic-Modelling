import pandas as pd 
import numpy as np 
import re
import pickle
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

import gensim

test_data = pd.read_csv(
    r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\test_data.csv"
)


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
    
dictionary = gensim.corpora.Dictionary(docs)

dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

texts = test_data.text.apply(clean)


lda_model = gensim.models.ldamodel.LdaModel.load(
    r"C:\Users\tejas\Documents\Deep Learning\My Work\Topic Modelling\LDA\lda_model"
)


topics = []
for i in range(len(texts)):
    bow_vector = dictionary.doc2bow(texts[i])
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        topics.append(lda_model.print_topic(index, topn=10))


topics = pd.Series(topics, dtype='category')

for i, v in enumerate(topics.cat.categories.tolist()):
    print(i, v)

topics = topics.cat.rename_categories(
    [
        'Contract and Pricing',
        'Customer Experience',
        'Payments and Billing',
        'Installation and Service Appointments',
        'Internet Speed and Connectivity'
    ]
)

print('\n')
print(topics.value_counts())

plt.bar(topics.value_counts().index, topics.value_counts().values)
plt.show()

plt.bar(topics.value_counts().index, 
        (topics.value_counts().values / topics.value_counts().values.sum() * 100))
plt.xlabel('Topics')
plt.ylabel('Percentage')
plt.title('Percentage of Different Types of Customer Complaints')
plt.show()

print(topics.value_counts().index, (topics.value_counts().values / topics.value_counts().values.sum() * 100))