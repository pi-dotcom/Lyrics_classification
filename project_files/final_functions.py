#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
my_text = pd.read_csv('all_lyrics.csv')
#my_text = my_text.set_index('artist')
my_text = my_text.drop(columns='Unnamed: 0')
#my_text


# In[12]:


raw_df = pd.read_csv('lyrics_final.csv')
#raw_df


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
X = raw_df['0']
y = my_text['artist']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


# In[15]:


def train_model(X_train, y_train):
    """
    Takes in list of songs
    trains model on it with labels,
    and returns trained model
    """
    print('\nTraining model...')
    #cv = CountVectorizer(stop_words='english')
    tf = TfidfVectorizer()
    #rf = RandomForestClassifier(max_depth=max_depth)
    clf = MultinomialNB()
    model = make_pipeline(tf, clf)
    model.fit(X_train, y_train)
    print('...and done!\n')
    return model


# In[16]:


def predict(pipeline, new_text):
    """
    Takes the pre-trained pipeline model and predicts new artist.
    """
    prediction = pipeline.predict(new_text)
    probs = pipeline.predict_proba(new_text)
    return prediction[0], probs.max()


# In[ ]:





# In[ ]:




