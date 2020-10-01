#!/usr/bin/env python
# coding: utf-8

# # Competencia Santander NLP

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ## 1 - Carga del Dataset

# In[2]:


df = pd.read_csv('data/train.csv',sep = '|')
df.columns = ['Pregunta', 'Intencion']


# ## 2 - Modelo Baseline

# ### Separacion Train y Test

# In[3]:


X = df.Pregunta
y = df.Intencion

X_train, X_test, y_train, y_test = train_test_split(df.Pregunta, df.Intencion, random_state = 0)


# ### Vectorización del texto utilizando CountVectorizer

# In[4]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer(sublinear_tf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# ### Entrenar un modelo Support Vector Machines

# In[5]:


clf = SVC(C=2)
clf.fit(X_train_tfidf, y_train)


# ### Predecir con los datos de Test

# In[6]:


X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
preds = clf.predict(X_test_tfidf)


# ### Metricas

# In[7]:


print('El valor de Accuracy en test es de: {}'.format(round(accuracy_score(y_test,preds),3)))

print('El valor de balanced Accuracy en test es de: {}'.format(round(balanced_accuracy_score(y_test,preds),3)))


# In[8]:


print(classification_report(y_test,preds))


# In[ ]:





# In[ ]:




