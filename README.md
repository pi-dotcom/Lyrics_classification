# Predicting the artist from web-scraped music lyrics

This project was completed in fourth week of the Data Science Bootcamp at Spiced Academy.

## Description
The goal is to build a web-scraper to scrape music lyrics, preprocess the obtained text, then predict the artist from some lyrics that a user writes in the command line. For testing, I scraped around 300 song lyrics of Linkin park, Coldplay and Breitney spears from [songlyrics.com](www.songlyrics.com) and used the texts to train a Multinomial Naive Bayes Classifier, which predicted the band from input lyrics with 68% accuracy.

- This program takes artist input from the user to train a Multinomial Naive Bayes model via parsing artists' songs from the website www.songlyrics.com
