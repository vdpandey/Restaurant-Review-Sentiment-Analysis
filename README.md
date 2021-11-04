# Restaurant-Review-Sentiment-Analysis


Any opinion/review given by any of an individual through which the feelings, text message, 
attitudes and thoughts can be expressed is known as the sentiment. The kinds of data analysis 
which are attained from the news reports, user reviews, social media updates or 
microblogging sites is called sentiment analysis which is also known as opinion mining. It is 
an approach which is used to analyse the sentiment of input data. The reviews of individuals 
towards certain events, brands, product or company can be known through sentiment analysis. 
The responses of the general public are collected and improvised by researchers to perform 
evaluations. The popularity of sentiment analysis is growing today since the numbers of views 
being shared by people on the microblogging sites are also increasing. All the sentiments can 
be categorized into three different categories called positive, negative and neutral. Twitter, 
being the most popular microblogging site, is used to collect the data to perform analysis. 
Tweepy is used to extract the source data from Twitter. Python language is used in this 
research to implement the classification algorithm on the collected data. The features are 
extracted using N-gram modelling technique. The sentiments are categorized among positive, 
negative and neutral using a supervised machine learning algorithm known as K-Nearest 
Neighbor.

Algorithms 
The data set chosen for this problem is available on Kaggle. The sentiment analysis is a 
classification because the output should be either positive or negative. That is why I tried 3 of 
the classification algorithms on this data set. 
● Multinomial Naive Bayes 
● Bernoulli Naive Bayes 
● Logistic Regression 
i) Multinomial Naive Bayes : Naive Bayes Classifier Algorithm is a family of 
probabilistic algorithms based on applying Bayes’ theorem with the “naive” assumption of 
conditional independence between every pair of a feature. Bayes theorem calculates 
probability P(c|x) where c is the class of the possible outcomes and x is the given instance 
which has to be classified, representing some certain features. 
P(c|x) = P(x|c) * P(c) / P(x) 
Naive Bayes is mostly used in natural language processing (NLP) problems. Naive Bayes 
predict the tag of a text. They calculate the probability of each tag for a given text and then 
output the tag with the highest one. 
ii) Bernoulli Naive Bayes BernoulliNB implements the naive Bayes training and
classification algorithms for data that is distributed according to multivariate 
Bernoulli distributions; i.e., there may be multiple features but each one is assumed to be a 
binary-valued (Bernoulli, boolean) variable. Therefore, this class requires samples to be 
represented as binary-valued feature vectors; if handed any other kind of data, a BernoulliNB 
instance may binarize its input (depending on the binarize parameter). 
14
The decision rule for Bernoulli naive Bayes is based on:- P ( x i ∣ y ) = P ( i ∣ y ) xi + ( 
1 − P ( i ∣ y ) ) ( 1 − x i )
which differs from multinomial NB’s rule in that it explicitly penalizes the non-occurrence of 
a feature that is an indicator for class, where the multinomial variant would simply ignore a 
non-occurring feature. 
In the case of text classification, word occurrence vectors (rather than word count vectors) 
may be used to train and use this classifier. BernoulliNB might perform better on some 
datasets, especially those with shorter documents. It is advisable to evaluate both models if 
time permits. 
iii) Logistic Regression is a supervised classification algorithm. In a classification 
problem, the target variable(or output), y, can take only discrete values for the given set 
of features(or inputs), X. 
Contrary to popular belief, logistic regression is a regression model. The model builds a 
regression model to predict the probability that a given data entry belongs to the category 
numbered as “1”. Just like Linear regression assumes that the data follows a linear function, 
Logistic regression models the data using the sigmoid function. 
 
15
 Framework 
Heroku- Heroku is a platform as a service (PaaS) that enables developers to build, run, and
operate applications entirely in the cloud. 
 Heroku architecture 
16
 Software and Hardware Requirements 
 4.1 SOFTWARE REQUIREMENTS 
● Operating system : Any operating system 
● Language : Python, HTML, CSS
 4.2 HARDWARE REQUIREMENTS 
● No specific requirement for hardware as the model is deployed on cloud hence only
the system with strong internet connectivity is required.
17
Designing 
The approach was straight forward. Few classifiers algorithms were selected for the project. I 
chose restaurant reviews as my project title. Firstly I understood the working of the algorithm 
and read about them. 
After gathering the data set from Kaggle. The first step was to process the data. In data 
processing, I used NLTK (Natural Language Toolkit) and cleared the unwanted words in my 
vector. I accepted only alphabets and converted it into lower case and split it in a list. Using 
the PorterStemmer method stem I shorten the lookup and Normalized the sentences. Then 
stored those words which are not a stopword or any English punctuation. 
Secondly, I used CountVectorizer for vectorization. Also used fit and transform to fit and 
transform the model. The maximum features were 1500. 
The next step was Training and Classification. Using train_test_split 30% of data was used for 
testing and remaining was used for training. The data were trained on all the 3 algorithms 
mentioned above. 
Later metrics like Confusion matrix, Accuracy, Precision, Recall were used to calculate the 
performance of the model. 
The best model was tuned to get a better result. 
Lastly, we checked the model with real reviews and found the model is detecting the
sentiments of the customer reviews properly. 
18
 Methodology 
All the models were judged based on a few criteria. These criteria are also recommended by 
the scikit-learn website itself for the classification algorithms. The criteria are: 
● Accuracy score: Classification Accuracy is what we usually mean when we use the 
term accuracy. It is the ratio of the number of correct predictions to the total number of 
input samples. 
● Confusion Matrix: A confusion matrix is a table that is often used to describe the 
performance of a classification model (or "classifier") on a set of test data for which 
the true values are known. i) There are two possible predicted classes: "yes" and "no". 
If we were predicting the presence of a disease, for example, "yes" would mean they 
have the disease, and "no" would mean they don't have the disease. ii) The classifier 
made a total of 165 predictions (e.g., 165 patients were being tested for the presence of 
that disease). iii) Out of those 165 cases, the classifier predicted "yes" 110 times, and 
"no" 55 times. iv) In reality, 105 patients in the sample have the disease, and 60 
patients do not. 
true positives (TP): These are cases in which we predicted yes (they have the disease), 
and they do have the disease. true negatives (TN): We predicted no, and they don't 
have the disease. 
false positives (FP): We predicted yes, but they don't have the disease. (Also known as 
a "Type I error.") 
false negatives (FN): We predicted no, but they do have the disease. (Also known as a 
"Type II error.") 
● F1 score is the Harmonic Mean between precision and recall. The range for F1 Score 
is [0, 1]. It tells you how precise your classifier is (how many instances it classifies 
correctly), as well as how robust it is (it does not miss a significant number of 
instances). High precision but lower recall, gives you an extremely accurate, but it 
then misses a large number of instances that are difficult to classify. The greater the F1 
Score, the better is the performance of our model. Mathematically, it can be expressed 
as: F1 Score tries to find the balance between precision and recall. 
● Precision: It is the number of correct positive results divided by the number of positive 
results predicted by the classifier. 
19
● Recall: It is the number of correct positive results divided by the number of all 
relevant samples (all samples that should have been identified as positive). 
