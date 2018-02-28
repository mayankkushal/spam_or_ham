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
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(output_dim=32, input_dim=5000, activation='relu', kernel_initializer="uniform"))
model.add(Dropout(0.9))
model.add(Dense(output_dim=64, activation='relu', kernel_initializer="uniform"))
model.add(Dropout(0.1))
model.add(Dense(output_dim=1, activation='sigmoid', kernel_initializer="uniform"))

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=75)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

model.save('ann_spam.h5')