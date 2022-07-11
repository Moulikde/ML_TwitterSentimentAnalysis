# TwitterSentimentAnalysis

The aim of this Natural Language Processing project is to develop Machine Learning Classifiers to classify text posts (tweets) from Twitter into Positive, Negative, or Neutral Labels based on the overall sentiment of the tweet.

#Project Goal and Methodology
The overall goal for this project was to use different NLP strategies to preprocess the tweet_content. We would then fit the data into different classificaiton models and compare the accuracy for each. 

#Non NLP Preprocessing
This includes removing any missing values and encoding both the sentiment (LabelEncoder) and the entities (OneHotEncoder) column. 

-------------------------------------------------------------------------------------------------------------------------------------------------------

#NLP Processing
Data Enrichment - We wanted to use a couple different methods to furthur enrich our dataset. We chose to use TextBlob to return the polarity and subjectivity of the tweet and Flair to measure the sentiment. 

Feature Extraction - The main NLP method used to convert the tweet_content into features was by using CountVectorizer (Bag of Words or BOW). We used the standard parameters for BOW with the exception of max_df, which we tuned. 

#Topic Modeling - Used Topic Modeling to extract headlines from tweet_content.


-------------------------------------------------------------------------------------------------------------------------------------------------------

#Classification Models

Model 1 - Multinomial NB - easy to build, can be applied to large datasets and can perform binary or multiclass classification. Very popular for text classification and sentiment analysis

Model 2 - Linear SVM - It is good for linear seperable problems, but could this algorithm perform when there are more than just two labels? We have Neutral and Irrelevant Sentiments in our dataset as well and by using linear SVM we can see how it handles these problems.

Model 3 - Polynomial SVM - Kernel - 'RBF' is known as the go-to kernel for non-linear problems. We wanted to see how a non-linear SVM contrasts with linear SVMs that's why wer decided to go with this model.

Model 4 - Neural Network - good for nonlinear problems such as sentiment analysis, also capable of multiclass or binary classification.

-------------------------------------------------------------------------------------------------------------------------------------------------------

#Citations

Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

Textblob: https://textblob.readthedocs.io/en/dev/

Flair: https://github.com/flairNLP/flair

Textblob: https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524

SVM Kernels: https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace

Kernels: https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02

-------------------------------------------------------------------------------------------------------------------------------------------------------

This is an academic group project in collaboration with [Eric Low](https://github.com/getlow012), a fellow data scientist with exceptional skill and talent in the domain.
