
1. INTRODUCTION 
News is the information about events that are provided through different media: social media, platforms, electronic communication, or through the testimonyof observers and
witnesses to events. The most important problem to be solved is to evaluate whether the data is mis-leaded or correct. 
Since fake news tends to spread faster than real news there is a need to classify news as fake or not. In the project, the dataset used is from the Kaggle website where real
news and fake news are in two separate datasets. We combined both datasets into one and trained with different machine learning classification algorithms to classify the news
as fake or not. In this project different feature engineering methods for text, data have been used like Tf-Idf model which is going to convert the text data into feature
vectors that are sent into machine learning algorithms to classify the news as fake or not. With different features and classification algorithms, we are going to classify
the news as fake or real. Algorithms and features which can give us the best result with that feature extraction method we are going to predict whether the news is fake or real.

In this project, we will be ignoring attributes like the id of the news, the text it contains, etc., and instead, focus only on the title and author of the article. We aim to 
use different machine learning algorithms and determine the best way to classify news.

2. LITERATURE SURVEY
2.1. Literature Review
In Today's world, everybody uses the internet to post content true or false over the internet. Unfortunately, counterfeit news gathers a lot of consideration over the web,
particularly via webbased networking media. Individuals get misdirected and don't reconsider before flowing such mis-educational pieces to the most distant part of the
arrangement. Such types of activities are not good for a society where some rumors or vague news evaporates the negative thought among the people or specific categories of
people[1]. As fast the technology is moving, at the same pace, preventive measures are required to deal with such activities. Broad communications assume a gigantic job in
impacting the general public and as it is normal, a few people attempt to exploit it. There are numerous sites that give false data. They deliberately attempt to bring out
purposeful publicity, deceptions, and falsehood under the presence of being true news. Their basic role is to control the data that can cause open to having confidence in it.
There are loads of cases of such sites everywhere throughout the world. Therefore, counterfeit news influences the brains of individuals. As indicated by studies Scientists
accept that numerous man-made brainpower calculations can help in uncovering bogus news.

The digital news industry in the United States is facing a complex future. On one hand, a steadily growing portion of Americans are getting news through the internet, many 
U.S. adults get news on social media, and employment at digital-native outlets has increased. On the other, digital news has not been immune to issues affecting the broader
media environment, including layoffs, madeup news, and public distrust. Half of Americans (52%) say they have changed the way they use social media because of the issue of 
made-up news. Furthermore, among the Americans who ever get news through social media, half have stopped following a news source because they thought it was posting made-up 
news and information. At the same time, about a third (31%) of social media news consumers say they at least sometimes click on news stories they think are made up. So, 
there is a need to stop fake news from spreading.

2.2. Research on Different Classification Algorithms used for Identifying Fake 
News 
2.2.1 Support Vector Machine 
Nicollas R. de Oliveira, Dianne S.V.Medeiros and Diogo M.F.Mattos developed a model for fake news detection [2]. They collected a fake news dataset from Boatos.org for detecting
fake news. In this, they have used a Support Vector Machine for detecting fake news. To achieve the goal they have used PCA and LSA methods for feature extraction. By using Support
Vector Machine they obtained an accuracy of 86%. Karishnu Poddar, Geraldine Bessie Amali D, and Umadevi K S developed a Comparison of Various Machine Learning Models for Accurate
Detection of Fake News [3]. They used a fake news dataset from kaggle.com for the detection of fake news by Naive Bayes Classifier, Logistic Regression, Decision Trees, Support 
Vector Machines, and Artificial Neural Networks. They used Count Vectorizer and TF-IDF Vectorizer feature extraction methods. SVM shows better results with TF-IDF with 92.8% accuracy.
Logistic regression performs equally for both count vectorizer and TF-IDF with accuracies of 91.6 and 91.0. Smitha, Bharat developed a Performance Comparison of Machine Learning 
Classifier for Fake News Detection [4]. They used Word Embedding for preprocessing of data. They used a Count vectorizer and TF-IDF for feature extraction. In their paper they 
used classifiers are Support Vector Machine, Logistic Regression, Decision Trees, Random Forest, XGBoost, and Gradient Boosting Neural Network for classifying the news as fake or 
real. By using different classification algorithms the highest accuracy obtained is with the SVM Linear classification algorithm with TF-IDF feature extraction with 94% accuracy. 

2.2.2 Logistic Regression 
It is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is 
a binary variable that contains data coded as 1 (yes and success) or 0 (no and failure). Iftikhar Ahmad, Muhammad Yousaf, Suhail Yousaf, and Muhammad Ovais Ahmad implemented a 
model for fake news detection on social media. They worked on Logistic Regression, SVM, and KNN models using social media and fake news datasets. LIWC method is used for feature
extraction. In their experiment, they found that Logistic Regression shows high accuracy of 91% compared to SVM 67%, and KNN-68%. 
Sruthi. M. S, Rahul R, Rishikesh G implemented An Efficient Supervised Method for Fake News Detection using Machine and Deep Learning Classifiers [5]. They used news channel data 
from Kaggle.com. In this paper, they used various methods like Naive Bayes, SVM, and LSTM. In this paper, they used stop-word removal, tokenization, sentence segmentation, a lower
casing, and punctuation removal for Preprocessing of data, and the tendency technique is used for feature extraction. Among the all methods LSTM gives more accuracy of 94.53%. 

2.2.3 Naive Bayes 
It uses probabilistic approaches and is based the on Bayes theorem. They deal with the probability distribution of variables in the dataset and predict the response variable of 
value. An advantage of a naive Bayes classifier is that only requires less bulk of training data to access the parameters necessary for classification. Mykhailo Granik and Volydimyr
Mesyura developed a Naive Bayes classifier technique for fake news detection. In this, they use Buzz feed news which contains information on Facebook content. In this, the 
classification accuracy for true is 75.59% and for false is 71.73% and the accuracy for total is 75.40%. Rahul M, Monica R, Mamathan N, and Krishana R developed a machine learning 
model for fake news detection by using FND-jru, Pontes Rout, and News Files datasets. After experimenting on different datasets, they found that each model shows variant accuracies 
on different datasets among them Naive Bayes, Passive Aggressive, and DNN gave better accuracies of 90%, 83%, and 80%. 

3. METHODOLOGY
3.1. Data Preprocessing 
There exploratory data analysis is performed on training data to prepare the data for modeling of systems like null or missing values, removing social media slang, removing stop-words,
and correcting contraction. Also, Part of Speech (PoS) Tagging has been performed in the data to meet the accuracy of the prediction model. Data has been stemmed to get the root
form of the words so that the prediction algorithm gets trained on the quality data. Before model training, the data was tokenized so that each word in the sentence can be treated
as an element for model training.

3.1.1 Stemming 
Stemming is one of the most common text pre-processing techniques used in Natural Language Processing (NLP) and machine learning in general. Stemming involves deriving the meaning
of a word from something like a dictionary. Stemming gives the root form of the word i.e., studying is studies. The root word is achieved by removing the suffix from a given word. 

3.1.2. Tokenization 
Tokenization is essentially splitting a phrase, sentence, paragraph, or entire text document into smaller units, such as individual words or terms. Each of these smaller units is
called a token. Here, tokens can be either word, characters, or sub-words as tokens are the building blocks of Natural Language, the most common way of processing the raw text 
happens at the token level. Hence, Tokenization is the foremost step while modeling text data. Tokenization is performed on the corpus to obtain tokens. The following tokens are 
then used to prepare a vocabulary. Vocabulary refers to the set of unique tokens in the corpus. Remember that vocabulary can be constructed by considering each unique token in
the corpus or by considering the top K Frequently Occurring Words.

3.2 Feature Selection 
In this module, we have performed feature selection methods from sci-kit learn python libraries. For feature selection, we have used methods like count Vectorization term frequency 
like tf-idf weighting.

3.2.1 Count Vectorization 
Vectorization is a process of converting text data into a machine-readable form. The words are represented as vectors. We cannot pass text directly to train our models in Natural
Language Processing, thus we need to convert it into numbers, which a machine can understand and can perform the required modeling on it. Countvectorizer tokenizer (tokenization
means breaking down a sentence or paragraph or any text into words) the text along with performing very basic preprocessing like removing the punctuation marks, converting all 
the words to lowercase, etc. The vocabulary of known words is formed which is also used for encoding unseen text later. An encoded vector is returned with a length of the entire
vocabulary and an integer count for the number of times each word appeared in the document. 

3.2.2 TF-IDF 
TF-IDF stands for “Term Frequency — Inverse Document Frequency”. This is a technique to quantify a word in documents, we generally compute a weight to each word which signifies the 
importance of the word in the document and corpus. This method is a widely used technique in Information Retrieval and Text Mining. TF is individual to each document and word,
hence we can formulate TF as follows. 
This measures the importance of documents in the whole set of the corpus, this is very similar to TF. The only difference is that TF is the frequency counter for a term t in 
document d, whereas DF is the count of occurrences of term t in the document set N. df(t) = occurrence of t in documents IDF is the inverse of the document frequency which measures 
the informativeness of term t. When we calculate IDF, it will be very low for the most occurring words such as stop words (because stop words such as “is” are present in almost all
of the documents, and N/df will give a very low value to that word). This finally gives what we want, a relative weightage. Tf-idf(t) = N/df 

3.3 Model Training 
The process of training an ML model involves providing an ML algorithm (that is, the learning algorithm) with training data to learn from. We can use the ML model to get predictions
on new data for which we do not know the target by training a model having the target variable in training data. The training data must contain a target or target attribute to know
the answer we want. The learning algorithm finds patterns in the training data that map the input data attributes to the target (the answer that you want to predict), and it 
outputs an ML model that captures these patterns. 

3.4 Model Evaluation 
Model evaluation is the process of using different evaluation metrics to understand a machine learning model’s performance, as well as its strengths and weaknesses. Model evaluation 
is important to assess the efficiency of a model during the initial research phases. The most popular metrics for measuring the Model’s performance include accuracy, precision, 
confusion matrix, Recall, F1-score, and AUC (area under the ROC curve). 
● Accuracy measures how often the classifier makes the correct predictions, as it is the ratio between the number of correct predictions and the total number of predictions. 
● Precision is the proportion of true results overall positive results. 
● Recall is the fraction of all correct results returned by the model. 
● F1-score is the weighted average of precision and recall between 0 and 1, where the ideal F1-scorevalue is 1.
