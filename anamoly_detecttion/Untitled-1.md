---
marp: true
theme: "Beige"
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
pd.options.plotting.backend= "plotly"
pd.set_option('display.max_columns', 150, 'display.max_rows', 100, 'display.max_colwidth', 15)
%matplotlib inline 
```
---
# Introduction 

* In the real world, fraud often goes undiscovered, and only the fraud that is caught provides any labels for the datasets. 

* Moreover, fraud patterns change over time, so supervised systems that are built using fraud labels become stale, capturing historical patterns of fraud but failing to adapt to newly emerging patterns.

* For these reasons (the lack of sufficient labels and the need to adapt to newly emerging patterns of fraud as quickly as possible), unsupervised learning fraud detection systems are in vogue.

* In this notebook, we will build such a solution using PCA 
--- 
# What is PCA

PCA (Principal Component Analysis) is a technique to find a low-dimensional representation of a dataset that captures as much variation as possible. It seeks a small number of dimensions that are interesting and informative, where each dimension is a linear combination of the original features. The first principal component is a normalized linear combination of the features that has the largest variance. It can be found through an optimization problem, and the resulting loadings and scores make up the principal component loading vector. PCA is useful when the original dataset has a large number of features, making it difficult to visualize and analyze.

---

# load dataset

```python
df = pd.read_csv('/Users/waleedidrees/Dropbox/Python_Projects/books/handson-unsupervised-learning-master/datasets/credit_card_data/credit_card.csv').rename(columns= {"Class":"target"})
df.head()
df.columns= df.columns.str.lower()
df.head()
```
---
# Disable the warnings
import warnings
warnings.filterwarnings('ignore')
df.describe().T

---

## we see that fraudulent transactions are very rare and this makes the data very imbalanced

df["target"].value_counts().reset_index()
we have 284,807 credit card transactions in total, of which 492 are fraudulent, with a positive (fraud) label of one. The rest are normal transactions, with a negative (not fraud) label of zero.
We have 30 features to use for anomaly detection—time, amount, and 28 principal components. And, we will split the dataset into a training set (with 190,820 transactions and 330 cases of fraud) and a test set
(with the remaining 93,987 transactions and 162 cases of fraud)

---
```python
(
df["target"]
.value_counts()
.reset_index()
.plot.bar(x="index", y= "target", color="index", height= 800, width= 800)
 )
```

---
```python
pd.options.plotting.backend = "matplotlib"

df.hist(figsize= (22,16), bins=50)

pd.options.plotting.backend = "plotly"
```

---