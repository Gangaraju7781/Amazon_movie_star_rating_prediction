#Import all libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

from sklearn import preprocessing
import scipy
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')

#Pre-processing
def process(df):
    # This is where you can do all your processing

    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    return df


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv") # (1697533, 10) all data

# Process the DataFrame
train_processed = process(trainingSet) # all data

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

#Exploratory Data Analysis
print(trainX.shape) # (1397533, 10) # after processing, remove all none row (testX)
print(submissionSet.shape) # (300000, 2) # Samples we need to predict
print(testX.shape) # (300000, 10)
trainX.describe()
trainX.head()

# Handling Missing Data
print(trainX.isnull().sum().sort_values())   
import missingno as msno
msno.bar(trainX,figsize=(6,3)) # bar chart
plt.show()

trainX['Text'].loc[trainX['Text'].isna()] = ''
trainX['Summary'].loc[trainX['Summary'].isna()] = ''
testX['Text'].loc[testX['Text'].isna()] = ''
testX['Summary'].loc[testX['Summary'].isna()] = ''

# replace missing data in the whole dataset
train_processed['Text'].loc[train_processed['Text'].isna()] = ''
train_processed['Summary'].loc[train_processed['Summary'].isna()] = ''


# EDA
# Most rated products
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
trainX['ProductId'].value_counts(sort=True).nlargest(15).plot.bar()
plt.title('15 Most Rated Products')

# Least rated products
plt.subplot(1,2,2)
trainX['ProductId'].value_counts(sort=True).nsmallest(15).plot.bar()
plt.title('15 least Rated Products')
plt.show()


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
trainX['HelpfulnessNumerator'].value_counts(sort=True).nlargest(15).plot.bar()
plt.title('Top 15 HelpfulnessNumerator')
plt.subplot(1,2,2)
trainX['Helpfulness'].value_counts(sort=True).nlargest(15).plot.bar()
plt.title('Top 15 Helpfulness (Numerator / Denominator)')
plt.show()


Numerator_0 = trainX[trainX['HelpfulnessNumerator'] == 0]['HelpfulnessNumerator'].value_counts()
Numerator_1 = trainX[trainX['HelpfulnessNumerator'] == 1]['HelpfulnessNumerator'].value_counts()
Numerator_2 = trainX[trainX['HelpfulnessNumerator'] == 2]['HelpfulnessNumerator'].value_counts()
Numerator_3 = trainX[trainX['HelpfulnessNumerator'] == 3]['HelpfulnessNumerator'].value_counts()
Numerator_4 = trainX[trainX['HelpfulnessNumerator'] > 3]['HelpfulnessNumerator'].value_counts()

labels = '0', '1', '2', '3', '> 3'
sizes = [Numerator_0.values.item(), Numerator_1.values.item(), Numerator_2.values.item(), Numerator_3.values.item(), Numerator_4.values.sum()]
explode = (0, 0, 0, 0, 0)
fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal') 
plt.title(' Pie Chart of Helpfulness Numerators')
plt.show()

# Rating
print(trainX['Score'].value_counts())
plt.figure(figsize=(9,5))
trainX['Score'].value_counts().plot.bar()
plt.title('Scores')
plt.show()

# analyze Helpfulness and Score, grouped by Score
plt.figure(figsize=(6,4))
print(trainX[['Helpfulness', 'Score']].groupby('Score').mean())
trainX[['Helpfulness', 'Score']].groupby('Score').value_counts(sort=True).nlargest(15).plot.bar()
plt.title('Top 15 Helpfulness Grouped by Scores')
plt.show()

# Natural language processing
trainX['TextLength'] = trainX.apply(lambda row: len(row['Text'].split(" ")) if type(row['Text']) == str else 0, axis = 1)
trainX['SummaryLength'] = trainX.apply(lambda row: len(row['Summary'].split(" ")) if type(row['Summary']) == str else 0, axis = 1)

SummaryList = pd.Series(trainX['Summary']).to_list()
TextList = pd.Series(trainX['Text']).to_list()
print(SummaryList[0])
print(TextList[0])

from sklearn.feature_extraction.text import CountVectorizer

# Assuming you have already created and fit your CountVectorizer object
vectorizer = CountVectorizer(analyzer='word', stop_words='english', lowercase=True, max_features=20)
summaryTopWords = vectorizer.fit_transform(SummaryList)

print("Top 20 Summary key words: ")
print(vectorizer.get_feature_names())

# Stem summary words
stemmer = SnowballStemmer('english')
new_list = []
for sentence in trainX['Summary']:
    tokens = []
    for word in nltk.word_tokenize(sentence):
        tokens.append(word)
    new_text = [stemmer.stem(token) for token in tokens]
    new_list.append(new_text)
  
print(new_text) 
print(len(new_text))

# Handling Time data
trainX['Date'] = pd.to_datetime(trainX['Time'], unit = 's')
trainX['Month'] = trainX['Date'].dt.month
trainX['Year'] = trainX['Date'].dt.year

# Handling Categorical Variables
ENC = preprocessing.OneHotEncoder()
data1 = ENC.fit_transform(train_processed[['ProductId', 'UserId']])
data1

# preprocessing for train part
text_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english')
data_text_vectors = text_vectorizer.fit_transform(train_processed['Text'])

data_summary_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english')
data_summary_vectors = data_summary_vectorizer.fit_transform(train_processed['Summary'])

# Feature Selection
train = trainX.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'TextLength','SummaryLength','Date','Month','Year'])
test = testX.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
train

# Modeling 
# split the data in train and test
mask = train_processed["Score"].isnull()
test_index = mask.to_numpy().nonzero()[0]
train_index = (~ mask).to_numpy().nonzero()[0]
train_X = scipy.sparse.csr_matrix(allFeature3)[train_index] 
test_X = scipy.sparse.csr_matrix(allFeature3)[test_index] 
train_X 

train_Y = train_processed['Score'].loc[train_processed['Score'].isna() == False]
test_Y = train_processed['Score'].loc[train_processed['Score'].isna()] # test for kaggle

# Score Lables for train and test
train_Y = train_Y.reset_index()['Score']
test_Y = test_Y.reset_index()['Score']
train_Y

# Dimensionality Reduction
dim_reduced_dataset = TruncatedSVD(25).fit_transform(train_X)
dim_reduced_dataset.shape

# svd decomposition
u, s, vt = np.linalg.svd(dim_reduced_dataset, full_matrices=False)

# plot
plt.plot(s)
plt.title('Singular Values')
plt.show()

from sklearn.decomposition import TruncatedSVD
train_X = TruncatedSVD(4).fit_transform(train_X)
test_X = TruncatedSVD(4).fit_transform(train_X)

# K-Nearest Neighbors
X_train, X_test, Y_train, Y_test = train_test_split(
        train_X, 
        train_Y, 
        test_size=1/4, 
        random_state=42
    )

# Learn the model
model = KNeighborsClassifier(n_neighbors=20).fit(X_train, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test)

# Evaluating model on test set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))
MSE = mean_squared_error(Y_test, Y_test_predictions) 
RMSE = np.sqrt(MSE) 
print("root mean-squared error = ", RMSE)

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Random Forest
X_train, X_test, Y_train, Y_test = train_test_split(
        train_X, 
        train_Y, 
        test_size=1/4, 
        random_state=42
    )

# Random Forest
model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X_train, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test)

# Evaluating model on test set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

MSE = mean_squared_error(Y_test, Y_test_predictions) 
RMSE = np.sqrt(MSE) #
print("root mean-squared error = ", RMSE)

cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Decision tree
X_train, X_test, Y_train, Y_test = train_test_split(
        train_X, 
        train_Y, 
        test_size=1/4, 
        random_state=42
    )

# From above, we set max_depth = 3
model = DTC().fit(X_train, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test)

# Evaluating model on test set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

MSE = mean_squared_error(Y_test, Y_test_predictions) 
RMSE = np.sqrt(MSE) #
print("root mean-squared error = ", RMSE)

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Load submission set
X_submission = pd.read_csv("./data/test.csv")
submission = pd.DataFrame(X_submission)
submission['Score'] = model.predict(test_X)

# Create the submission file
submission.to_csv("./data/submission.csv", index=False)
