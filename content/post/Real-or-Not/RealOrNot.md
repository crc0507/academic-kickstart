---
date: 2017-12-01
title: Real or Not? NLP with Disaster Tweets
---

### 该博客说明了如何通过逻辑回归来预测哪些推文与实际灾难有关，哪些与逻辑灾难无关。
### link to kaggle:  
https://www.kaggle.com/ruochenchang/kernel5ed82c5423
### This Notebook was ranked: 1739 and achieved an accuracy of 0.79447 on the test data
### 1. Importing necessory libraries


```python
import numpy as np 
import pandas as pd 

# text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords

# sklearn 
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/nlp-getting-started/test.csv
    /kaggle/input/nlp-getting-started/sample_submission.csv
    /kaggle/input/nlp-getting-started/train.csv


### 2. Load training data and test data. 

Training data has 5 columns. Test data has 4 columns. Test did not have the target feature because we should predict the target value for the test data.


```python
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train_data.head() #dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test_data.head() #dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>



2.1 Learn the missing values in training data and test data.

It seems keywords and location values are missing in both training data and test data.
And location values missed a lot.


```python
train_data.info()
test_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7613 entries, 0 to 7612
    Data columns (total 5 columns):
    id          7613 non-null int64
    keyword     7552 non-null object
    location    5080 non-null object
    text        7613 non-null object
    target      7613 non-null int64
    dtypes: int64(2), object(3)
    memory usage: 297.5+ KB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3263 entries, 0 to 3262
    Data columns (total 4 columns):
    id          3263 non-null int64
    keyword     3237 non-null object
    location    2158 non-null object
    text        3263 non-null object
    dtypes: int64(1), object(3)
    memory usage: 102.1+ KB



```python
train_data.isnull().sum()
```




    id             0
    keyword       61
    location    2533
    text           0
    target         0
    dtype: int64




```python
test_data.isnull().sum()
```




    id             0
    keyword       26
    location    1105
    text           0
    dtype: int64



2.2 In the training data, 42% of the target value is 1 and others are 0.


```python
train_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7613.000000</td>
      <td>7613.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5441.934848</td>
      <td>0.42966</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3137.116090</td>
      <td>0.49506</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2734.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5408.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8146.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10873.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data['target'].value_counts()
```




    0    4342
    1    3271
    Name: target, dtype: int64



### 3. Data pre processing

3.1 Data cleaning




```python
def clean_text(text):
    # Make text lowercase, remove punctuation.
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

# Applying the cleaning function to both test and training datasets
train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))
test_data['text'] = test_data['text'].apply(lambda x: clean_text(x))

train_data['text'].head()
```




    0    our deeds are the reason of this earthquake ma...
    1                forest fire near la ronge sask canada
    2    all residents asked to shelter in place are be...
    3    13000 people receive wildfires evacuation orde...
    4    just got sent this photo from ruby alaska as s...
    Name: text, dtype: object



3.2 Tokenization
 Divide the text into words which can also be called tokens.


```python
# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train_data['text'] = train_data['text'].apply(lambda x: tokenizer.tokenize(x))
test_data['text'] = test_data['text'].apply(lambda x: tokenizer.tokenize(x))
train_data['text'].head()
```




    0    [our, deeds, are, the, reason, of, this, earth...
    1        [forest, fire, near, la, ronge, sask, canada]
    2    [all, residents, asked, to, shelter, in, place...
    3    [13000, people, receive, wildfires, evacuation...
    4    [just, got, sent, this, photo, from, ruby, ala...
    Name: text, dtype: object



3.3 Remove stopwords


```python
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

train_data['text'] = train_data['text'].apply(lambda x : remove_stopwords(x))
test_data['text'] = test_data['text'].apply(lambda x : remove_stopwords(x))
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[deeds, reason, earthquake, may, allah, forgiv...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[forest, fire, near, la, ronge, sask, canada]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[residents, asked, shelter, place, notified, o...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[13000, people, receive, wildfires, evacuation...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[got, sent, photo, ruby, alaska, smoke, wildfi...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



3.4 Normalization


```python
# After preprocessing, the text format
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

train_data['text'] = train_data['text'].apply(lambda x : combine_text(x))
test_data['text'] = test_data['text'].apply(lambda x : combine_text(x))
train_data['text']
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>deeds reason earthquake may allah forgive us</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>forest fire near la ronge sask canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>residents asked shelter place notified officer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13000 people receive wildfires evacuation orde...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>got sent photo ruby alaska smoke wildfires pou...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Vectorization


```python
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_data['text'])
test_vectors = count_vectorizer.transform(test_data["text"])

print(train_vectors[0].todense())
print(test_vectors.shape[1])

```

    [[0 0 0 ... 0 0 0]]
    22223


### 5. Build a classification model

Logic regression classifier


```python
# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(clf, train_vectors, train_data["target"], cv=5, scoring="f1")
scores
```




    array([0.59865255, 0.48421053, 0.56658185, 0.5540797 , 0.68765133])




```python
clf.fit(train_vectors, train_data["target"])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv('my_submission.csv', index=0)
print("success")
```

    success

