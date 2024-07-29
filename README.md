# Twitter-NLP-Sentiment-Analysis

This project analyzes tweets and the existence of racist/sexist content in these tweets. This project utilized Logistic Regression to predict the existence of racist/sexist language in different tweets collected. 

Using Bag-of-Words and TF-IDF feature extraction techniques, our model shows to have a slightly higher accuracy with TF-IDF. This is perhaps due to the higher frequencies of positive words on Twitter, allowing TF-IDF to perform more appropriately. 

First, I cleaned the data by removing short words, twitter handles, and converting words to their roots. After doing so, I performed EDA by plotting wordclouds and bar graphs of common positive vs. negative hashtags. 
As shown below, here are some common positive vs. negative words within tweets. 
Positive:
![Wordcloud](https://github.com/user-attachments/assets/03eeae7b-90b3-40b6-b0d4-b2243e245995)
Negative:
![Negative Wordcloud](https://github.com/user-attachments/assets/f4ad74f5-2679-4ae7-a9b8-531d0b79e47c)

Here are bar graphs of common positive hashtags vs. negative hashtags.
![Positive Hashtags](https://github.com/user-attachments/assets/1bbbb6a1-1436-4be9-b22b-5c9cd8cddee6)
![Negative Hashtags](https://github.com/user-attachments/assets/f1b8fa2d-3c2e-4ab2-8ebb-203818420ff4)

After comparing positive and negative tweets, I started building a Logistic Regression model using both the Bag-of-Words (BoW) and TF-IDF feature extraction techniques. I split the dataset into 70/30 training/testing datasets. I also set a prediction threshold of 0.3, meaning the prediction is considered correct if the probability is greater than or equal to 30%. This is because the dataset contains significantly more positive tweets than negative tweets. 

After training the model, we obtained a F1 score of 0.53 using the BoW technique and a F1 score of 0.54 using the TF-IDF technique. I am assuming that TF-IDF performs slightly better due to the high frequencies of positive/non-negative words in the data, thus applying less weights on them and more weights on the negative words.
The F1 score is calculated by: F1 Score = 2*(Recall * Precision) / (Recall + Precision), where:
True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
False Positives (FP) – When actual class is no and predicted class is yes.
False Negatives (FN) – When actual class is yes but predicted class in no.

Precision = TP/TP+FP
Recall = TP/TP+FN

Overall, this model performed decently. In the future, other regression models or machine learning techniques could be assessed to compare their accuracies with the current model. 
