import spacy
import pandas as pd
import numpy as np
from spacy.vectors import Vectors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

nlp= spacy.load('en_core_web_lg')
df=pd.read.csv('yelp.csv')
print(df.head())
print(df.shape)
df.head()
print(df.head)
sentence_vector = [nlp(x).vector for x in df[ 'text'].values]
sentence_vector=np.stack( sentence_vector, axıs=0)
print(sentence_vector.shape)
svd = TruncatedSVD (n_components=10)
svd_sentences=svd.fit_transform(sentence_vector)
ny_sentece_vec = np.stack([nlp("This is best class").vector])
sentence_vector=np.append (sentence_vector, ny_sentece_vec, axis=0)
print(sentence_vector.shape)
svd_sentences = svd.fit_transform(sentence_vector)
cos_sin=cosine_similarity(svd_sentences, svd_sentences)
degisken2=pd.DataFrane(cos_sin) [457].sort_values(ascending= False) [:10]
print("DEGISKEN2 DEGERI")
print (degisken2)
print("This is World class.", '\n')
print(df['text'][419])
print(svd_sentences)
cos_sin=cosine_similarity (svd_sentences, svd_sentences)
degisken=pd.DataFrame(cos_sin) [18].sort_values (ascending=False) [:10]
print("data frame")
print (degisken)
print("değişken altındaki")
print (df[ 'text'][28],'\n')
print(df['text'][2174])