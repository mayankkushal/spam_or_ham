# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

# Cleaning the texts

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    sms = word_tokenize(dataset['v2'][i])
    ps = PorterStemmer()
    sms = [ps.stem(word.lower()) for word in sms if not word in set(stopwords.words('english'))]
    sms = ' '.join(sms)
    corpus.append(sms)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=7000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = GaussianNB()

#Fitting Random Forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', 
                                    n_jobs = -1, max_depth=10)

# Fitting and Predicting the Test set results
classifier.fit(X_train, y_train)
y_pred_prob = classifier.predict_proba(X_test)

y_pred = (y_pred_prob[:, 1] > 0.184).astype(float)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)


#Plotting ROC curve
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Saving the classifier model
import pickle
filename = 'finalized_model_spam.sav'
pickle.dump(classifier, open(filename, 'wb'))


