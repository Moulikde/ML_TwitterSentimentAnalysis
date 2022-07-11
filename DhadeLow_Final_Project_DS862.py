#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:11:12 2022

@author: ericlow
"""

import pandas as pd
import numpy as np
import seaborn as sns

#Project Goal and Methodology
#The overall goal for this project was to use different NLP strategies to preprocess the tweet_content. We would then fit the data into different classificaiton models and compare the accuracy for each. 

#Non NLP Preprocessing - This includes removing any missing values and encoding both the sentiment (LabelEncoder) and the entities (OneHotEncoder) column. 

#NLP Processing
##Data Enrichment - We wanted to use a couple different methods to furthur enrich our dataset. We chose to use TextBlob to return the polarity and subjectivity of the tweet and Flair to measure the sentiment. 

##Feature Extraction - The main NLP method used to convert the tweet_content into features was by using CountVectorizer (Bag of Words or BOW). We used the standard parameters for BOW with the exception of max_df, which we tuned. 

#Topic Modeling

#Classification Models Selected

    ##Model 1 - Multinomial NB - easy to build, can be applied to large datasets and can perform binary or multiclass classification. Very popular for text classification and sentiment analysis

    ##Model 2 - Linear SVM - It is good for linear seperable problems, but could this algorithm perform when there are more than just two labels? We have Neutral and Irrelevant Sentiments in our dataset as well and by using linear SVM we can see how it handles these problems.
    ##Model 3 - Polynomial SVM - Kernel - 'RBF' is known as the go-to kernel for non-linear problems. We wanted to see how a non-linear SVM contrasts with linear SVMs that's why wer decided to go with this model.

    ##Model 4 - Neural Network - good for nonlinear problems such as sentiment analysis, also capable of multiclass or binary classification. 

#Model Setup
    ##In addition to comparing the accuracy for each classification model, we also wanted to compare the                        accuracy of a model with just the tweet_content as a feature vs a model that utilizes ALL the features (including enrichment) with the exception of Neural Network. 
    ##) Split dataset into train/valid/test. All models will use the same splits 60%/20%/20%

    ##Model Methodology
    #1) Model1a/2a/3a - tweet_content feature only
        ### Use RandomizedGridSearch to tune best hyperparameters for BOW and classification model using train_valid dataset. 
        ### Measure accuracy against test set
        
    #2 Model1b/2b/3b - All features (including enrichment)
        ### Using the best hyperparameter for BOW found in the previous model, apply bow.fit_transform to the tweet_content column and save as sparse matrix. 
        ### Drop tweet_content column from train_valid dataset, convert to sparse matrix
        ### Combine both sparse matrices together to form new sparse matrix
        ### Fit the specified classification model with this sparse matrix and tune hyperparameters using RandomizedGridsearch
        ### Test Dataset
            ####Transform using the bow.transform function
            ####drop tweet_content column and convert to sparse matrix
            ####combine both sparse matrix to form new test sparse matrix
        ###Measure accuracy against test sparse matrix using best hyperparameters from the classification model
    #3 Model 4 - All features (including enrichment)
        ###Similar to Model1b/2b/3b, use best bow hyperparameters from model1 to convert tweet_content and combine with train_valid sparse matrix
        ###Transform y_train_valid using TensorFlow's to_categorical and convert to sparse matrix
        ###Fit NN with train_valid sparse matrix and tune hyperparameters using RandomizedGridSearch
        ###Convert X_test using same method as Model1b/2b/3b
        ###Measure accuracy against test sparse matrix using best hyperparameters from the classification
        
#Compare accuracies 
#%% Preprocessing

#import data
columns = ['tweet_id', 'entity', 'sentiment', 'tweet_content']
df = pd.read_csv('twitter_training.csv', names = columns)

#look for missing values
df.info()
print(df.isnull().sum())
df.isnull().sum().sort_values(ascending = False).plot(kind='bar') #686 missing values

#In the tweet_content column, there were some rows that had a blank space that needed to be removed
df['tweet_content'].replace(' ', np.nan, inplace = True)
df.dropna(axis=0,inplace = True)
df.reset_index(inplace = True, drop = True)
df.info()

#Encode Sentiment using LabelEncoder
from sklearn.preprocessing import LabelEncoder
labenc = LabelEncoder()
df['sentiment'] = labenc.fit_transform(df['sentiment'])

#Encode Entity using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
#Fit transform Entity
hot_entity = hot.fit_transform(df[['entity']])
#Create DF of encoded data
hot_entity_df = pd.DataFrame(hot_entity.todense(), columns = hot.get_feature_names())
#Combine DF
Full_df = pd.concat([df, hot_entity_df], axis = 1)

#Drop unnecessary columns
Full_df.drop(['entity', 'tweet_id'], axis = 1, inplace = True)

print(Full_df)

# %%% Textblob
#One of the two methods we'll use to enrich the data is to apply TextBlob on the tweet_content column. TextBlob is a python library that uses the Natural Language Toolkit (NLTK) that allows users to perform functions such as tokenization, noun phrase extration, translation, classificatio and even sentiment analysis. For this project, we just want to take advantage of texdtblob's ability to return the polarity and subjectivity of a sentence (or tweet in our case) and add them as features to our dataset. 

#The polarity is a value that is between [-1, 1] and provides the sentiment for the tweet. A value of -1 is negative, +1 is positive and 0 is neutral. 

#The subjectivity refers to how much of the tweet can be classified as an opinion or a matter of fact. This value ranges [0, 1], with a value of 0 being very objective (fact) and 1 is very subjective (opinion). 

#Below, we also use a minmaxscaler on textblob's polarity values to get away from negative values that may cause issues with fitting our models

#pip install textblob
from textblob import TextBlob

#Create function to add polarity and subjectivity to df
def textblob_polarity(tweet_content):
    """returns polarity from TextBlob"""
    
    polarity = []
    
    for tweet in tweet_content:
        testimonial = TextBlob(tweet)
        polarity.append(testimonial.sentiment[0])

    return polarity

def textblob_subjectivity(tweet_content):
    """returns subjectivity from TextBlob"""
    
    subjectivity = []
    
    for tweet in tweet_content:
        testimonial = TextBlob(tweet)
        subjectivity.append(testimonial.sentiment[1])
    
    return subjectivity

Full_df['txtblb_polarity']= textblob_polarity(Full_df['tweet_content'])
Full_df['txtblb_subjectivity'] = textblob_subjectivity(Full_df['tweet_content'])

#use minmaxscaler on txtblb polarity to get away from negative values
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler()
Full_df['txtblb_polarity'] = mmscaler.fit_transform(Full_df[['txtblb_polarity']])

#%%% Flair
#The 2nd method we'll use to enrich our data is called Flair. Flair is a powerful NLP library that can perform a variety of NLP functions such as named entity recognition (NER), part of speech tagging (PoS), classification and can also support multiple languages. Understanding the context of a word is one of the biggest reasons why Flair is gaining popularity. We'll use Flair to return the predicted sentiment and add it to our dataset. For the sentiment, Flair will simply return a POSITIVE or a NEGATIVE. The confidence will return a range [0,1] and indicates how confident the sentiment analysis is. 



#pip install flair

from flair.models import TextClassifier
from flair.data import Sentence

#examples
classifier = TextClassifier.load('en-sentiment')
test = df['tweet_content'].loc[18]
print(test)
sentence = Sentence(test)
classifier.predict(sentence)
sentence.labels[0]

#create function to add sentiment and confidence to df
def flair_sentinment_confidence(tweet_contents):
    """returns sentiment and confidence from Flair"""
    sentiment = []
    confidence = []
    
    for tweet in tweet_contents:
        testimonial = Sentence(tweet)
        classifier.predict(testimonial)
        
        #append lists
        sentiment.append(testimonial.labels[0].value)
        confidence.append(testimonial.labels[0].score)
    return sentiment, confidence

Full_df['flair_sentiment'], Full_df['flair_confidence'] = flair_sentinment_confidence(Full_df['tweet_content'])

#encode flair sentiment using LabelEncoder
Full_df['flair_sentiment'] = labenc.fit_transform(Full_df['flair_sentiment'])

#%% Optional Save
#Applying Flair and Textblob on the dataset does take quite awhile (especially for Flair) so it's recommended to save the file at this point.

#optional save and reload
Full_df.to_csv('full_df.csv', index = False)

#load csv
data = pd.read_csv('full_df.csv', error_bad_lines=False)



#%% Dataframe Sample and Assign X and y
#small sample
data = data.sample(15000)
data.shape #(15000, 38)

X = data.copy()
y = X.pop('sentiment')


#%% Data Exploration
X_HM = pd.concat([X, y], axis = 1)

#Sentiment Distribution
Sentiment = labenc.inverse_transform(y)
ax = sns.countplot(x=Sentiment, data=X_HM, palette='deep')
#Observation: The distributions of the 4 sentiment types are not exactly evenly distributed. There's about 1.5X more negative sentiment compared to irrelevant sentiment. Neutral and Positive fairly close (probably differ by about 500 observations) while the sentiment with the most observations is negative. 

#Entity distribution
ax = sns.catplot(y="entity", kind = 'count', data=df, palette='deep')
#Observation: Fairly equal distribution between all entity types

#%%Topic Modeling
#We will use Topic Modeling to scan the tweet_content column in our dataset, it will scan the tweet_content (documents), detect word patterns within these tweets and cluster the word groups that would essentially summarize the documents.

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

# Create a DF
data = pd.read_csv('full_df.csv', error_bad_lines=False)

# We only need the tweet_content column from the data
data_text = data['tweet_content'] #We are using the entire dataset

# To use the first 30000 lines
#data_text = data['tweet_content'][:30000]

# Check number of documents
print("\n",'----------------------------------------------------',"\n")
print("Our dataset has", len(data_text),"documents")
print("\n",'----------------------------------------------------',"\n")
print(data_text.head())
print("\n",'----------------------------------------------------',"\n")
#Our dataset has 73824 documents

# In[ ]:

#We chose these parameters because they were the best parameters for one of our classification models (MNB), these parameters are appropriate for 32 entities/topics in our dataset.
# Use Bag of Words to find Headlines matrix
bow = CountVectorizer(stop_words = 'english', min_df = 1, max_df = 0.13870000000000138)
documents_bow = bow.fit_transform(data_text)
documents_bow.shape

# We have 73824 headlines and 13807 unique words in our dataset

# In[ ]:

# Fit the LDA with 32 topics because we have 32 entities in our dataset
num_topics = 32
lda = LatentDirichletAllocation(n_components = num_topics, random_state = 862)
lda_results = lda.fit_transform(documents_bow)

print("\n",'----------------------------------------------------',"\n")
print(lda.components_) 
print("\n",'----------------------------------------------------',"\n")
print(lda.components_.shape)
print("\n",'----------------------------------------------------',"\n")
print(lda_results.shape)
print("\n",'----------------------------------------------------',"\n")

# In[ ]:

# Create a dataframe for components (It can also be viewed as distribution over the words for each topic after normalization)
topics = pd.DataFrame(lda.components_, columns=bow.get_feature_names())
topics.head(10)


# In[ ]:
    
# Assign a Dataframe to lda_results to vizualize the results better
LDAResults = pd.DataFrame(lda_results, columns = ['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4','Topic5', 
                                     'Topic6', 'Topic7', 'Topic8', 'Topic9', 'Topic10', 'Topic11',
                                     'Topic12', 'Topic13', 'Topic14', 'Topic15', 'Topic16', 'Topic17',
                                     'Topic18', 'Topic19', 'Topic20', 'Topic21', 'Topic22', 'Topic23'
                                     'Topic24', 'Topic25', 'Topic26', 'Topic27', 'Topic28', 'Topic29',
                                     'Topic30', 'Topic31', 'Topic32'])

LDAResults.head(10)

# In[ ]:

# Display the top 20 words for each of the 32 topics
def show_topics(word_topic_matrix, word_labels, num_top_words=20):
    top_words_func = lambda x: [word_labels[i] for i in np.argsort(x)[:-num_top_words-1:-1]]
    topic_words = ([top_words_func(i) for i in word_topic_matrix])
    return [' '.join(x) for x in topic_words]

# Use the function on the LDA result
show_topics(lda.components_, bow.get_feature_names())


# In[ ]:
#Top 10 words in top 6

#Topic 0: 'johnson baby powder com talc selling based stop cancer canada company products https vaccine market sales news just 2020 lawsuits',
#Topic 1: 'home verizon depot italy phone ve service just work people customer homedepot today month want going money worst com days',
#Topic 2: 'warcraft world just look amazing game like fucking did hell youtu love smh video good year don doing want old',
#Topic 3: 'youtube com watch banned youtu fifa people 20 pubg video chinese new 2020 channel wow nvidia apps man amazing twitter',
#Topic 4: 'wtf com good microsoft disappointed new congratulations win help cs 2020 great giveaway make 10 feeling nr wn https man',
#Topic 5: 'twitter pic com rhandlerr facebook thank love new just want fun years ve people google team fm lot update news',

   
# In[ ]:

# Calculate k

scores = []
for k in range(1, 32):
    lda = LatentDirichletAllocation(n_components=k, random_state = 862)
    lda.fit(documents_bow) # Fit the model
    scores.append(lda.score(documents_bow)) # Obtain the loglikelihood score

print("\n",'----------------------------------------------------',"\n")
print(scores)
print("\n",'----------------------------------------------------',"\n")
print(np.argmax(scores))
print("\n",'----------------------------------------------------',"\n")
#30
    
# In[ ]:

# Fit & Transform using Tfidf
tfidf = TfidfVectorizer(stop_words='english', min_df=1, max_df = 0.13870000000000138)
documents_tfidf = tfidf.fit_transform(data_text)
documents_tfidf.shape
#(73824, 30764)

# In[ ]:
  
# Let's run NMF on the data
nmf = NMF(n_components = num_topics, max_iter = 1000, init = 'nndsvd') # We will use 'nndsvd' as initialization method
W = nmf.fit_transform(documents_tfidf)
H = nmf.components_

# Again let's look at the words for each topic
show_topics(H, tfidf.get_feature_names_out())

print("\n",'----------------------------------------------------',"\n")
print(W[0:5])
print("\n",'----------------------------------------------------',"\n")

#Top words in top 5 topic

#Topic 0: 'game eamaddennfl fix nba2k rainbow6game played ve ghostrecon make servers ass years broken year trash playing madden worst update ubisoft',
#Topic 1:  'unk pubg happy nba2k real duty better welcome smh eamaddennfl bro com youtu lol cool nba2k_myteam ya facebook got stop',
#Topic 2:  'pic twitter com rhandlerr facebook pubg google wikipedia org overwatch banned fm fortnite fix youtube years 2020 thanks microsoft account',
#Topic 3:  'love world new warcraft guys gta playing games absolutely people borderlands god omg way fortnite guy list duty pubg boy',
#Topic 4:  'dead red redemption redeeming rockstargames games online story finished things gta com masterpiece rockstar played amazing youtu ve time redem',

# In[ ]:

#Word cloud generates a dictionary of top_words with (top 50 words) based on their respective weights.    

from wordcloud import WordCloud

def topics_freq_dict(topic_world_matrix, word_labels, num_top_words=20):
    top_words_func = lambda x: [(word_labels[i] ,x[i]) for i in np.argsort(x)[:-num_top_words-1:-1]]
    topic_words_freq = ([top_words_func(i) for i in topic_world_matrix])
    tuple_output = [x for x in topic_words_freq]
    list_dict_word_freq = []
    
    for topic in tuple_output:
        dict_output = {}
        for word_freq in topic:
            dict_output[word_freq[0]] = word_freq[1]
            
        list_dict_word_freq.append(dict_output)
        
    return list_dict_word_freq

list_dict_topics = topics_freq_dict(lda.components_, bow.get_feature_names(), 50) #list of dictionaries for each topic's top 50 words
print(list_dict_topics)

# Create the word clouds
for dict_topics in list_dict_topics:
    wordcloud = WordCloud(background_color="white", height=1000, width=2000).generate_from_frequencies(dict_topics)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

#%% Data Splits - Final Train/Valid/Test splits are 60%/20%/20%
from sklearn.model_selection import train_test_split

#Split 80% train_valid, 20% test
X_train_valid, X_test, y_train_valid, y_test  = train_test_split(X, y, test_size= 0.2, random_state=5)

#Split train_valid into 60% train, 20% valid
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size= 0.25, random_state = 5)

#%%% Model 1A: Bag-Of-Words + MultinomialNB - use tweet_content feature only
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy
from scipy import sparse

#Setup RandomizedGridsearchCV to tune hyperparameters
pipe = Pipeline([('CV', CountVectorizer(stop_words = 'english')),
                     ('mnb', MultinomialNB())])

paras = {'CV__max_df': np.arange(.09, .14, .0001), 
         'mnb__alpha': np.logspace(-10, 0, 11)}

model1a = RandomizedSearchCV(pipe, paras, cv = 10, n_jobs = -1, verbose = 1, scoring = 'accuracy', n_iter = 160, random_state = 5)
model1a.fit(X_train_valid['tweet_content'], y_train_valid)
model1a_best_params = model1a.best_params_
print(model1a_best_params)

#Prediction against Test Set
y_pred = model1a.predict(X_test['tweet_content'])
model1a_acc = accuracy_score(y_pred, y_test)
print(model1a_acc) #80.827% (for 20,000 rows of data, so with more data we got even better accuracies.)

# Precision Scores Vizualized for each label
Model = 'MNB'
PS = list((precision_score(y_test, y_pred, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
M1APSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=M1APSDF)

model1a_acc = accuracy_score(y_test,y_pred) #66.53% (for current 15,000 rows of data we obtained 66.53% accuracy.)

#%%%% Result: Accuracy on test set : 66.53%

#%%% Model 1B: Bag-Of-Words + MultinomialNB - use all features
#Convert tweet_content using BOW. Use tuned parameters from model1a
cv_model1b = CountVectorizer(stop_words = 'english', 
                     max_df = model1a.best_params_['CV__max_df'])

bow_model1b = cv_model1b.fit_transform(X_train_valid['tweet_content'])

#make copy of X_train_valid and drop tweet_content
X_train_valid_cv = X_train_valid.copy()
X_train_valid_cv.drop('tweet_content', axis = 1, inplace = True)

#Convert X_train_valid df to sparse df 
X_train_valid_cv_sparse = scipy.sparse.csr_matrix(X_train_valid_cv.values)

#Combine BOW with X_train_valid
X_train_valid_model1b_final = hstack((bow_model1b, X_train_valid_cv_sparse))

#Fit RandomizedSearchCV to tune hyperparameters
pipe = Pipeline([('mnb', MultinomialNB())])
paras = {'mnb__alpha': np.logspace(-5, 0, 6)}

model1b = RandomizedSearchCV(pipe, paras, cv = 10, n_jobs = -1, verbose = 1, scoring = 'accuracy', n_iter = 160, random_state = 5)
model1b.fit(X_train_valid_model1b_final, y_train_valid)
model1b_best_params = model1b.best_params_
print(model1b_best_params)

#Transform X_test
#Convert tweet_content using BOW
X_test_bow_model1b = cv_model1b.transform(X_test['tweet_content'])
#make copy of X_test and drop tweet_content
X_test_cv = X_test.copy()
X_test_cv.drop('tweet_content', axis = 1, inplace = True)
#Convert X_test df to sparse df
X_test_cv_sparse = scipy.sparse.csr_matrix(X_test_cv.values)
#combine BOW with X_test
X_test_model1b_final = hstack((X_test_bow_model1b, X_test_cv_sparse))

#Predict against test set
y_pred = model1b.predict(X_test_model1b_final)
model1b_acc = accuracy_score(y_pred, y_test)
print(model1b_acc) #81.20% (for 20,000 rows of data)

# Precision Scores Vizualized for each label
Model = 'MNB_Full'
PS = list((precision_score(y_test, y_pred, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
M1BPSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=M1BPSDF)

model1b_acc = accuracy_score(y_test,y_pred) #68.03% (Obtained on current 15,000 rows of data.)

#%%%% Result: Accuracy on test set : 68.03%
#%% Support Vector Machine
#%%%SVM - Model 2a - Linear SVC + Bag of words - use tweet_content feature only
from sklearn.svm import LinearSVC

#Setup RandomizedGridsearchCV to tune hyperparameters
pipe = Pipeline([('CV', CountVectorizer(stop_words = 'english')), 
                 ('clf', LinearSVC(random_state = 5, max_iter = 1000000, tol = 1e-10))])

paras = {'CV__max_df': np.arange(.4, .5, .01),
         'clf__C':[i for i in np.linspace(.05, .2, 11)],
         'clf__loss': ['hinge','squared_hinge']}


model2a = RandomizedSearchCV(pipe, paras, cv = 10, n_jobs = -1, verbose = 1, scoring = 'accuracy', n_iter = 160, random_state = 5)
model2a.fit(X_train_valid['tweet_content'], y_train_valid)
model2a_best_params = model2a.best_params_
print(model2a_best_params) 

#Prediction against Test Set
y_pred = model2a.predict(X_test['tweet_content'])
model2a_acc = accuracy_score(y_pred, y_test)
print(model2a_acc)

# Precision Scores Vizualized for each label
Model = 'LinSVC'
PS = list((precision_score(y_test, y_pred, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
M2APSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=M2APSDF)

model2a_acc = accuracy_score(y_test,y_pred) #68.3%

#%%%% Result: Accuracy on test set : 68.3%

#%%%Model 2b - Linear SVC + Bag of Words - use all features
#Convert tweet_content using BOW. Use tuned parameters from model2a
cv_model2b = CountVectorizer(stop_words = 'english', 
                             max_df = model2a.best_params_['CV__max_df'])

bow_model2b = cv_model2b.fit_transform(X_train_valid['tweet_content'])

#use previously created X_train_valid sparse matrix
X_train_valid_cv_sparse

#Combine BOW with X_train_valid
X_train_valid_model2b_final = hstack((bow_model2b, X_train_valid_cv_sparse))


#Fit RandomizedSearchCV to tune hyperparameters
pipe = Pipeline([('clf', LinearSVC(random_state = 5, max_iter = 1000000, tol = 1e-10))])

params = {'clf__C':[i for i in np.linspace(.05, .2, 11)],
          'clf__loss': ['hinge','squared_hinge']}

model2b = RandomizedSearchCV(pipe, params, cv = 10, n_jobs = -1, verbose = 1, scoring = 'accuracy', n_iter = 160, random_state = 5)

model2b.fit(X_train_valid_model2b_final, y_train_valid) 

#Best params
model2b_best_params = model2b.best_params_
print(model2b_best_params,'\n')

#Transform X_test
#Convert tweet_content using BOW
X_test_bow_model2b = cv_model2b.transform(X_test['tweet_content'])

#use X_test sparse matrix
X_test_cv_sparse

#combine BOW with X_test
X_test_model2b_final = hstack((X_test_bow_model2b, X_test_cv_sparse))

#predict on test set
y_pred = model2b.predict(X_test_model2b_final)
print("-------------------------------------------------------------------------",'\n')
print("Predicted values:", y_pred,'\n')
print("-------------------------------------------------------------------------",'\n')
print("Accuracy on the test set:", accuracy_score(y_test,y_pred),'\n')
print("-------------------------------------------------------------------------",'\n')
print("Classification report:\n", classification_report(y_test, y_pred),'\n')

# Precision Scores Vizualized for each label
Model = 'LinSVC_Full'
PS = list((precision_score(y_test, y_pred, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
M2BPSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=M2BPSDF)

model2b_acc = accuracy_score(y_test,y_pred) #68.55%

#%%%% Result: Accuracy on test set : 68.55% (on 10,000 datapoints)

# %%% SVM - Model 3A - Polynomial SVC + Bag of words - use tweet_content feature only

from sklearn.svm import SVC
#Fit RandomizedSearchCV to tune hyperparameters
pipe = Pipeline([('CV', CountVectorizer(stop_words = 'english')),
                     ('clf', SVC(random_state = 5))])


params = {'CV__max_df': [i for i in np.linspace(.15, .23, 5)], 
          'clf__C': [i for i in np.linspace(.1, .9, 5)],
          'clf__kernel': ['rbf', 'sigmoid'],
          'clf__degree': [i for i in np.linspace(.25, .33, 5)],
          'clf__coef0': [i for i in np.linspace(.05, .13, 5)]}

model3a = RandomizedSearchCV(pipe, params, cv = 10, n_jobs = -1, verbose = 1, scoring = 'accuracy', n_iter = 160, random_state = 5)

model3a.fit(X_train_valid['tweet_content'], y_train_valid)
model3a_best_params = model3a.best_params_
print(model3a_best_params,'\n')

#Prediction against Test Set
y_pred = model3a.predict(X_test['tweet_content'])
model3a_acc = accuracy_score(y_pred, y_test)
print(model3a_acc) #65.2%

# Precision Scores Vizualized for each label
Model = 'SVC'
PS = list((precision_score(y_test, y_pred, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
M3APSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=M3APSDF)

model3a_acc = accuracy_score(y_test,y_pred) #65.2%

#%%%% Result: Accuracy on test set : 65.2%

# %%% SVM - Model 3b - SVC + Bag of words - use all features

#Convert tweet_content using BOW. Use tuned parameters from model2a
cv_model3b = CountVectorizer(stop_words = 'english', 
                     max_df = model3a.best_params_['CV__max_df'])

bow_model3b = cv_model3b.fit_transform(X_train_valid['tweet_content'])

#use previously created X_train_valid sparse matrix
X_train_valid_cv_sparse

#Combine BOW with X_train_valid
X_train_valid_model3b_final = hstack((bow_model3b, X_train_valid_cv_sparse))

#Classifier
pipe = Pipeline([('clf', SVC(random_state = 5))])

params = {'clf__kernel': ['rbf', 'sigmoid'],
          'clf__degree': [i for i in np.linspace(.25, .33, 5)],
          'clf__coef0': [i for i in np.linspace(.05, .13, 5)]}

model3b = RandomizedSearchCV(pipe, params, cv = 10, n_jobs = -1, verbose = 1, scoring = 'accuracy', n_iter = 160, random_state = 5)

model3b.fit(X_train_valid_model3b_final, y_train_valid)
model3b_best_params = model3b.best_params_
print(model3b_best_params,'\n')

#Transform X_test
#Convert tweet_content using BOW
X_test_bow_model3b = cv_model3b.transform(X_test['tweet_content'])

#use X_test sparse matrix
X_test_cv_sparse

#combine BOW with X_test
X_test_model3b_final = hstack((X_test_bow_model3b, X_test_cv_sparse))

#predict on test set
y_pred = model3b.predict(X_test_model3b_final)

print("-------------------------------------------------------------------------",'\n')
print("Predicted values:", y_pred,'\n')
print("Accuracy on the test set:", accuracy_score(y_test,y_pred),'\n')
print("-------------------------------------------------------------------------",'\n')
print("classification report:\n", classification_report(y_test, y_pred),'\n')

# Precision Scores Vizualized for each label
Model = 'SVC_Full'
PS = list((precision_score(y_test, y_pred, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
M3BPSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=M3BPSDF)

#Accuracy
model3b_acc = accuracy_score(y_test,y_pred) #67.5%

#%%%% Result: Accuracy on test set : 67.5%

#%% Neural Network

#%%% Preprocessing
#Preprocess X_train_valid
#use BOW hypertuned parameters from model1a
cv = CountVectorizer(stop_words = 'english', max_df = model1a.best_params_['CV__max_df'])
X_tv_bow = cv.fit_transform(X_train_valid['tweet_content'])
X_tv_bow = X_tv_bow.todense()
X_tv_bow = pd.DataFrame(X_tv_bow, index = X_train_valid.index)

#drop tweet_content from X_train_valid
X_tv_c = X_train_valid.copy()
X_tv_c.drop('tweet_content', axis = 1, inplace = True)
X_train_valid_final = pd.concat([X_tv_c, X_tv_bow], axis = 1, sort = False)


#Preprocess X_test
X_test_bow = cv.transform(X_test['tweet_content'])
X_test_bow = X_test_bow.todense()
X_test_bow = pd.DataFrame(X_test_bow, index = X_test.index)

#drop tweet_content from X_test
X_test_c = X_test.copy()
X_test_c.drop('tweet_content', axis = 1, inplace = True)
X_test_final = pd.concat([X_test_c, X_test_bow], axis = 1, sort = False)

#convert to sparse matrix
X_train_valid_finalS = scipy.sparse.csr_matrix(X_train_valid_final.values)
X_test_finalS = scipy.sparse.csr_matrix(X_test_final.values)

#Categorically Encode y_train_valid (not needed for binary classification)
from tensorflow.keras.utils import to_categorical
y_train_valid_cat = to_categorical(y_train_valid, len(y_train_valid.unique()))

#Categorically Encode y_test (not needed for binary classification)
from tensorflow.keras.utils import to_categorical
y_test_cat = to_categorical(y_test, len(y_test.unique()))

#%%% Set seed to get reproducible results
from tensorflow.random import set_seed
set_seed(5)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

def build_model(n_hidden = 1, 
                n_neurons = 30, 
                input_dim = X_train_valid_finalS.shape[1], 
                activation = 'relu', 
                learning_rate = 0.001, 
                optimizer = 'adam'):
    
    #Instantiate the model
    model = Sequential()
    
    #Setup hidden layers
    options = {"input_dim": input_dim} # Set options 
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation = 'relu', **options)) # Here we are using the input options from before
        options = {} # Now we erase the input options so it won't be included in future layers
        
    #Setup output layer & optimizer
    model.add(Dense(4, activation = 'softmax')) #use activiation = sigmoid if performing binary classification
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    model.compile(loss = 'categorical_crossentropy', metrics = 'accuracy', optimizer = optimizer) 
    return model

keras_cl = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model) 

paras = {'n_hidden': [i for i in range (9, 22, 3)],
        'n_neurons':[i for i in range (30, 51, 5)]}

nn_rs= RandomizedSearchCV(keras_cl, paras, n_jobs = -1, cv = 10 , n_iter=160, scoring = 'accuracy', verbose = 1)

nn_rs.fit(X_train_valid_finalS, y_train_valid, epochs =100,
          callbacks = tf.keras.callbacks.EarlyStopping(patience=3))

print(nn_rs.best_params_)
#{'n_neurons': 36, 'n_hidden': 13}


#Measure Accuracy on Test Set
nn_results = nn_rs.predict(X_test_finalS)

#this step is not needed for binary classification
nn_results = to_categorical(nn_results, len(pd.Series(nn_results).unique())) 

#save accuracy results
nn_acc = accuracy_score(nn_results, y_test_cat)
print(nn_acc)
# Precision Scores Vizualized for each label
Model = 'NN'
PS = list((precision_score(y_test_cat, nn_results, average=None)))
d = {'labels': [0, 1, 2, 3], 'Sentiment': ['Negative', 'Neutral', 'Irrelevant', 'Positive'], 'Precision Score': PS, 'Model': Model}
NNPSDF = pd.DataFrame(data=d)
ax = sns.barplot(x="Sentiment", y="Precision Score", data=NNPSDF)

#%%%% Result: Accuracy on test set : 64.42%

#%%% Final Results

#Precision Scores for each label in each Model

#Concat all the PS Dataframes
pdList = [M1APSDF,M1BPSDF,M2APSDF,M2BPSDF,M3APSDF,M3BPSDF,NNPSDF]  # List of your dataframes
new_df = pd.concat(pdList)

#Concat all the PS Dataframes
sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=new_df, kind="bar",
    x="Model", y="Precision Score", hue="Sentiment",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Model", "Precision Score")
g.legend.set_title("")


# In[ ]:

#Plot Accuracy Score bar chart
a = {'Model':['MNB', 'MNB_Full', 'LinSVC', 'LinSVC_Full', 'SVC', 'SVC_Full', 'NN'], 'Accuracy':[model1a_acc, model1b_acc, model2a_acc, model2b_acc, model3a_acc, model3b_acc, nn_acc]}
ax = sns.barplot(x="Model", y="Accuracy", data=a)

#Observations: The best performing model ended up being model2b. We were very surprised that the neural network did not end up performing better. However, we were not surprised that the models that used the full features ended up performing better than just using the tweet_content alone. 

#%%%Final Remarks
#1. One of the major blunders we made in this project was not making sure the unique sentiment values in the database (positive, negative, neutral, irrelevant) matched up with the sentiment that was returned from Flair (positive or negative). The consequence is that an observation that had an ‘irrelevent’ or ‘neutral’ sentiment for example, would always be mislabeled by Flair. This most likely negatively affected our accuracy score. We did conduct a test using 15000 observations from the full_df with only positive and negative sentiment analysis. The results are below

#model1a (MultinomialNB + BOW, tweet_content only): 87.03%
#model1b (MultinomialNB + BOW, all features) : 87.86%
#model2a (Linear SVC + BOW, tweet_content only): 88.96%
#model2b (Linear SVC + BOW, all features): 89.7%
#model3a (Polynomial SVC + BOW, tweet_content only): 88.83%
#model3b (Polynomial SVC + BOW, all features)): 88.2%
#model4 (NN + BOW parameters from model1a, all features): 91.67%

#As you can see, the accuracy is much higher that our previous tests. We previously observed significant increases in accuracy as the observations in the test dataset increased and we expect the parallels here. The increase in accuracy could be attributed to the Flair Sentiment playing a more significant role as well as decreasing from multiclass to binary classification. 

#If you'd like to see this for yourself, simple run the code below followed by each model. 
#For model4, the y values need to be in a different format which is commented out in the NN section. 

test = data[(data['sentiment'].isin([1, 3]))] #Sentiment Classification: negative = 1, positive = 3
test.sample(15000) #This value can be changed to increase the sample size
X = test.copy()
y = X.pop('sentiment')

#2. We did not realize the magnitude, processing power and time needed to process all 75K obvervations in the full dataset. For example, if we applied BOW with model1a's tuned hyperparameters, we ended up with a train_valid dataset of size (59059, 29451). We continued to reduce the datset down to 50K and 30K, but even with a reduced sample size of 20K observations, we still had a train_valid dataset of size (16000, 20247). We did try a test with 500 observations to make sure our code ran correctly, but the the datasize is far too small to make any concrete observations about the full dataset. This is why we had to continuously reduce the dataset down to a manageable size where 1) our laptop would not run out of memory and 2) all 7 models could be completed in a reasonable time. 

#3. Finally, for models 1b/2b/3b, we used the tuned hyperparameter for BOW from the 'a' models which may not be the best candidate value to use for this model as the datasets are different (tweet_content vs all features). We could not figure out a method that could partially tune BOW using just the tweet_content while simulatenously tune the specific classification model using the other features. Thus, the next best option was to split off the tweet_content from the train_valid features and transform it using the tuned BOW hyerparameter from the 'a' model. Afterwards, we would combine this sparse matrix with the rest of the features to form a new sparse matrix to tune the classification model. 

#4. A future project could be to analyze the same dataset and compare accuracy using TextBlob, Flair and BOW since all 3 types are capable of performing sentiment analysis.

#5. 'RBF' is the better kernel for non-linear classification problems, we realized later that using sigmoid as one of our candidates wasn't ideal. Using Linear SVM would yield better results if we had only 0s and 1s but we have more labels so we could have used other better models for these problems. But then again, on doing a binary classification later linear SVM was the second best performing model in our project.

#Citations
#Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
#Textblob: https://textblob.readthedocs.io/en/dev/
#Flair: https://github.com/flairNLP/flair
#Textblob: https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524
#SVM Kernels: https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace
#Kernels: https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02

