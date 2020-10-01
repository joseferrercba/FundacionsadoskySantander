#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import nltk
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from googletrans import Translator
from classes.ModelBuilder import ModelBuilder
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample, shuffle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, balanced_accuracy_score
from classes.CustomTokenizer import *
pd.set_option('display.max_colwidth', 600)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#### Get Info from CSV

# In[2]:


columns = ['Pregunta', 'Intencion']
df_train = shuffle(pd.read_csv('data/train.csv', usecols=columns, sep='|'))
df_train['Intencion_cat_label'] = df_train['Intencion'].str[4:]
df_train['Intencion_cat_label'] = df_train['Intencion_cat_label'].astype('int32')


# #### Helper Code
# In[3]:
def plot_occurrences(data, title='Dummy Title'):
    plt.figure(figsize=(35,6))
    sns.countplot(x=data['Intencion_cat_label'], data=data, alpha=0.8, order=data['Intencion_cat_label'].value_counts().index)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Intencion', fontsize=2)
    plt.xticks(rotation=90)
    plt.title(title, fontsize=20)
    plt.show()
#### Category Analisis
# In[4]:
plot_occurrences(df_train)
# In[5]:
grouped = df_train.groupby('Intencion_cat_label')['Intencion'].count().sort_values(ascending=True)
# In[6]:
major_samples_ids = grouped[(grouped.values < grouped.max()) & (grouped.values > 50)].index.sort_values(ascending=True)
mid_samples_ids = grouped[(grouped.values <= 50) & (grouped.values > 20)].index.sort_values(ascending=True)
low_samples_ids = grouped[(grouped.values <= 20) & (grouped.values > 0)].index.sort_values(ascending=True)
print('Size each group: Major:{}, Mid:{}, Low:{}'.format(len(major_samples_ids), len(mid_samples_ids), len(low_samples_ids)))
# In[7]:
major_samples = df_train[df_train['Intencion_cat_label'].isin(major_samples_ids)]
plot_occurrences(major_samples, 'Major Samples (between < {} & > {} Occurrences)'.format(grouped.max(), 50))
mid_samples = df_train[df_train['Intencion_cat_label'].isin(mid_samples_ids)]
plot_occurrences(mid_samples, 'Mid Samples (between <= 50 & > 20 Occurrences)')
low_samples = df_train[df_train['Intencion_cat_label'].isin(low_samples_ids)]
plot_occurrences(low_samples, 'Low Samples (between <= 20 & > 0 Occurrences)')
# #### Translations
# In[8]:
df_train['Preguntas_custom_preprocess_no_stopwords'] = [custom_preprocess(sentence, conjugate_verbs=False, removeStopWords=False) for sentence in tqdm.tqdm(df_train['Pregunta'], ascii=True)]
df_train['Preguntas_custom_preprocess'] = [custom_preprocess(sentence, conjugate_verbs=False) for sentence in tqdm.tqdm(df_train['Pregunta'], ascii=True)]
df_train['Preguntas_custom_preprocess_w_verbs'] = [custom_preprocess(sentence, conjugate_verbs=True) for sentence in tqdm.tqdm(df_train['Pregunta'], ascii=True)]
df_train['Preguntas_word_tokenize'] = [nltk.word_tokenize(sentence) for sentence in tqdm.tqdm(df_train['Pregunta'], ascii=True)]
# In[9]:
poor_values = df_train[df_train.Intencion_cat_label == 132]
poor_values.sample(3)
# In[27]:
from wordcloud import WordCloud
preguntas = [listToString(sentence) for sentence in poor_values.Preguntas_word_tokenize]
text = " ".join(review for review in pd.unique(preguntas))
# Generate a word cloud image
wordcloud = WordCloud(background_color='black',                      
                      max_words=300,
                      width=800,
                      height=300).generate_from_text(text)
# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# In[30]:
from wordcloud import WordCloud
preguntas = [listToString(sentence) for sentence in df_train.Preguntas_custom_preprocess]
text = " ".join(review for review in pd.unique(preguntas))
# Generate a word cloud image
wordcloud2 = WordCloud(background_color='black',                      
                      max_words=300,
                      width=800,
                      height=300).generate_from_text(text)
# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")


# In[11]:
translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr'])
sentences = df_train.Pregunta.values.tolist()
questions = []
for sentence in tqdm.tqdm(sentences, ascii=True):
    translation = translator.translate(text=sentence, src='es', dest='en')    
    questions.append(translation.text)
df_train['Pregunta_en'] = questions

mode='w'
header=True
df_train.to_csv('data/train_preprocessed.csv',mode=mode, header=header, index=False, sep='|')


# In[13]:


# translations_es_back = []
# for sent in tqdm(translations_fr):
#     translation = translator.translate(sent, src="fr", dest="es").text
#     translations_es_back.append(translation)
# print(f'Amount sentences en: {len(translations_es_back)}')
# translations_es_back[:2]
# 
# df["Pregunta_es"] = translations_es_back


# In[14]:


#tokenizer = CustomTokenizer()
#words = [tokenizer.processAll(sentence, stem=False) for sentence in df_train['Pregunta']]
#
#freq_dist = nltk.FreqDist(np.concatenate(words, axis=0))
#freq_df = pd.DataFrame(list(freq_dist.items()), columns = ["Word","Frequency"])
#
#print('Coincidences for 1: {}'.format(len(freq_df[freq_df['Frequency'] == 1])))
#print('Coincidences for 2: {}'.format(len(freq_df[freq_df['Frequency'] == 2])))
#print('Coincidences for 3: {}'.format(len(freq_df[freq_df['Frequency'] == 3])))
#print('Coincidences for more than 3: {}'.format(len(freq_df[freq_df['Frequency'] > 3])))
#print('FreqDist')
#print(freq_df.sort_values(by='Frequency', ascending=True))

