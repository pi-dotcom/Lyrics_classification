# Predicting the artist from web-scraped music lyrics

This project was completed in fourth week of the Data Science Bootcamp at Spiced Academy.

## Description
The goal is to build a web-scraper to scrape music lyrics, preprocess the obtained text, then predict the artist from some lyrics that a user writes in the command line. For testing, I scraped around 300 song lyrics of Linkin park, Coldplay and Breitney spears from [songlyrics.com](www.songlyrics.com) and used the texts to train a Multinomial Naive Bayes Classifier, which predicted the band from input lyrics with 68% accuracy.

- This program takes artist input from the user to train a Multinomial Naive Bayes model via parsing artists' songs from the website www.songlyrics.com

I have used following concepts(lessions) for this project
-Web Scraping, Regular Expression, HTML Parsing, Language Models, Class Imbalance, Bag-of-Words, Naive Bayes, Python Functions, Command Line Interface
## Workflow of Project

### Web Scrapping
The first step in this project was to build a web-scraper in Python with Regular Expression and BeautifulSoup. I found this to be the most difficult and frustrating part, from finding a good lyrics website that doesn’t contain tens of duplicate lyrics, to implementing BeautifulSoup, and transforming the code from a JupyterNotebook into a Python file.

### Text Processing
Next, I had to clean and preprocess the scraped lyrics, in order to feed them to the classification model. I used SpaCy to tokenize the text, make the words lowercase, remove punctuation, numbers, and stop words (i.e. filling words or words that don’t change the meaning of a sentence, like the, a, to). This step was quite easy, mainly because the lyrics were in English.

### Naive Bayes Classifier
Now that that the dataset was made of clean lyrics, it was ready to train a classification model. I used the Multinomial Naive Bayes Classifier (MNBC), a probabilistic model based on Bayes’ Theorem, which assumes strong independence between the features and uses a multinomial distribution for each of the features. MNBC is typically used in text classification for calculating the probability of a word occurring in a text. For this, the words are vectorized or transformed into numbers. My model achieved an accuracy of 85% on the train set and 79% on the test set.
