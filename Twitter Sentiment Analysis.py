import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings 
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.chdir('/Users/stellenshun.li/Desktop/Project/Twitter NLP')

#Set training testing datasets
train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

#3 columns (id, label, and tweet)
#train.head()

###Remove twitter handles
#Combine train and test for convenience
combi = pd.concat([train, test], ignore_index=True)

#Function to remove unwanted patterns
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

#Creates new column with removed twitter handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")


###Remove special characters, numbers, punctuations except hashtags
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ", regex=True)

###Remove words shorter than 3 or less
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#print(combi.head())

###Split strings of texts into tokens (splits on spaces)
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
#print(tokenized_tweet.head())

###Stemming: removes suffixes like 'ing', 'ed', 'ly', etc
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
 # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet

#------------------------------------------------------------------------------------------------------#

###EDA with cleaned data
#Wordcloud to see most common words
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")co
plt.axis('off')
plt.show()

#Now common words in non racist/sexist tweets
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Lastly common words in racist/sexist tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Find trends in hashtags
#Function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i) #"\w+" extracts the full hashtag rather than detecting the presence of a hashtag
        hashtags.append(ht)
    return hashtags

#Extract hashtags from non-racist/sexist tweets
ht_normal = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

#Extract hashtags from racist/sexist tweets
ht_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

#Unnesting list
ht_normal = sum(ht_normal,[])
ht_negative = sum(ht_negative, [])

#Plot hashtags in non-racist/sexist tweets
a = nltk.FreqDist(ht_normal)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
#select top 10 most frequent hashtags
d = d.nlargest(columns="Count", n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel = 'Count')
plt.show()

#Plot hashtags in racist/sexist tweets
b = nltk.FreqDist(ht_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()),
                  'Count': list(b.values())})
#select top 10 most frequent hashtags
e = e.nlargest(columns="Count", n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

#------------------------------------------------------------------------------------------------------#

### Extracting features from the cleaned tweets
#Bag-of-words features: used to find commonality between tweets
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')

#bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

#TF-IDF features
    #penalizes the common words by assigning them lower weights while giving importance to words 
    #which are rare in the entire corpus but appear in good numbers in few documents
#TF = (# of times term t appears in a document)/(# of terms in the document)
#IDF = log(N/n), where N is the # of documents and n is the # of documents a term t has appeared in
#TF-IDF = TF*IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer =  TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
#TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


### Model building: sentiment analysis
#build the predictive models on the dataset using the 2 feature set (bag-of-words and TF-IDF)
#Use logistic regression: predicts the probability of occurrence of an event by fitting data to a logit function

#Building model using Bag-of-Words features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

#splits data into training and testing datasets
xtrain_bow, xtest_bow, ytrain, ytest = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

#Trains the Logistic Regression model
logistic = LogisticRegression()
logistic.fit(xtrain_bow, ytrain) 

prediction = logistic.predict_proba(xtest_bow)
#Threshold of 0.3 because the model is more sensitive (it predicts positive more oftenprediction_int = prediction_int.astype(int)

print(f"F1 score for Bag-of-Words: {f1_score(ytest, prediction_int)}")

#Use model to predict on testing dataset
test_pred = logistic.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) 

#Now we try TF-IDF features
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xtest_tfidf = train_tfidf[ytest.index]

logistic.fit(xtrain_tfidf, ytrain)

prediction = logistic.predict_proba(xtest_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)

print(f"F1 score for TF-IDF: {f1_score(ytest, prediction_int)}")