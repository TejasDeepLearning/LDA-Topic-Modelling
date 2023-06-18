# Topic Modelling with Streamlit

This repository contains code for performing topic modelling using Latent Dirichlet Allocation (LDA) and visualizing the results using Streamlit.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
The purpose of this code is to perform topic modelling on a collection of documents using the LDA algorithm and visualize the topics and their probabilities using Streamlit. The code takes a user input text and assigns it to one of the pre-defined topics based on the trained LDA model.

## Installation
To run this code locally, follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies by running the following command:

`pip install -r requirements.txt`

3. Download the required NLTK corpora by running the following command:

`python -m nltk.downloader stopwords`
`python -m nltk.downloader wordnet`


## Usage
1. Run the `main.py` script using the following command:

`streamlit run app.py`

2. Once the Streamlit app is running, you will see a text area where you can enter your text.
3. Enter the text you want to classify into one of the pre-defined topics.
4. The app will clean the text by removing special characters, stopwords, and applying lemmatization and stemming techniques.
5. The cleaned text will be converted into a bag-of-words vector using a pre-defined dictionary.
6. The LDA model will assign probabilities to each topic based on the bag-of-words vector.
7. The app will display the most probable topic and its corresponding probability.
