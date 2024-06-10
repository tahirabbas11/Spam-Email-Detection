# Spam Email Detection

This project aims to detect spam emails using machine learning techniques. It uses a logistic regression model to classify emails as spam or not spam based on their content.

## Dependencies

To get started, make sure you have the following dependencies installed:

```python
import numpy as np # to perform a wide variety of mathematical operations on arrays
import pandas as pd # to structure data from CSV
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer # to convert text data to numeric data for ML
from sklearn.linear_model import LogisticRegression # logistic regression model
from sklearn.metrics import accuracy_score # to evaluate model performance
```

## Data Collection & Pre-Processing
Loading the data from CSV file to a pandas DataFrame:
```python
raw_mail_data = pd.read_csv('/content/mail_data.csv')
print(raw_mail_data)
```
Replacing null values with an empty string:
```python
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
```
Printing the first 5 rows of the DataFrame:
```python
mail_data.head()
```

Checking the number of rows and columns in the DataFrame:
```python
mail_data.shape
```

## Label Encoding

Label spam mail as 0 and ham mail as 1:

```python
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
```
Separating the data into texts and labels:

```python
X = mail_data['Message']
Y = mail_data['Category']
print(X)
print(Y)
```
## Splitting the Data

Splitting the data into training data and test data:
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
```
##Feature Extraction

Transforming the text data to feature vectors:
```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```
Converting Y_train and Y_test values to integers:
```python
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
```
## Training the Model
Training the Logistic Regression model with the training data:
```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```
## Evaluating the Trained Model
Prediction on training data:
```python
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)
```
Prediction on test data:
```python
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)
```
## Building a Predictive System
Making predictions on new data:
```python
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
# input_mail=["PRIVATE! Your 2004 Account Statement for 07742676969 shows 786 unredeemed Bonus Points. To claim call 08719180248 Identifier Code: 45239 Expires"]

# Convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# Making prediction
prediction = model.predict(input_data_features)
print(prediction)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```



