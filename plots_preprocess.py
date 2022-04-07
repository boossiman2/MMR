#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import csv
import numpy as np
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[3]:


class Plots_preprocessor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.embedding_dict = dict()

    def read_csv(self):
        f = open('./data/del_plots.csv', 'r', encoding='utf-8')
        csvreader = csv.reader(f)
        next(csvreader, None)
        doc_list = []
        plot_list = []

        for movie_info in csvreader:
            try:
                index, m_Id, movie, plot = movie_info
            except ValueError:
                continue
            if plot == '':
                continue
            doc_list.append([m_Id, movie, plot])
            plot_list.append(plot)
        print(len(doc_list),len(plot_list))
        return doc_list, plot_list

    def stop_words(self, txt_list):
        norm_plots = []
        for sentence in txt_list:
            norm_plot = re.sub(r"[^a-z0-9]+", ' ', sentence.lower())
            if 'read all' in norm_plot:
                norm_plot = norm_plot.replace('read all', '')   
            norm_plots.append(norm_plot)
        return norm_plots
    
    def tokenizing(self, normalized_plot):
        self.tokenizer.fit_on_texts(norm_plot)
        vocab_size = len(self.tokenizer.word_index) + 1
        x_encoded = self.tokenizer.texts_to_sequences(norm_plot)
        max_len = max(len(i) for i in x_encoded)
        return x_encoded, max_len, vocab_size
    
    def padding(self, x_encoded, maxlen, padding):
        X_train = pad_sequences(x_encoded, maxlen=max_len, padding='post')
        
    def embedding_glove(self, vocab_size):
        f = open('./embedding/glove.6B.100d.txt', encoding="utf8")
        for line in f:
            word_vector = line.split()
            word = word_vector[0]
            # 100개의 값을 가지는 array로 변환
            word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
            self.embedding_dict[word] = word_vector_arr
        f.close()
        print('%s개의 Embedding vector가 있습니다.' % len(self.embedding_dict))
        
        glove_embedding_matrix = np.zeros((vocab_size, 100))
        print('임베딩 행렬의 크기(shape) :',np.shape(glove_embedding_matrix))
        
        for word, index in self.tokenizer.word_index.items():
        # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
            vector_value = self.embedding_dict.get(word)
            if vector_value is not None:
                glove_embedding_matrix[index] = vector_value
        return glove_embedding_matrix
    
    def embedding_word2vec(self, vocab_size):
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./embedding/GoogleNews-vectors-negative300.bin.gz', binary=True)
        print('모델의 크기(shape) : ', word2vec_model.vectors.shape)
        word2vec_embedding_matrix = np.zeros((vocab_size, 300))
        print('임베딩 행렬의 크기(shape) :',np.shape(word2vec_embedding_matrix))
        def get_vector(word):
            if word in word2vec_model:
                return word2vec_model[word]
            else:
                return None
        for word, index in self.tokenizer.word_index.items():
            # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
            vector_value = get_vector(word)
            if vector_value is not None:
                word2vec_embedding_matrix[index] = vector_value
        return word2vec_embedding_matrix


# In[ ]:


# MAIN


# In[4]:


preprocessor = Plots_preprocessor()


# In[5]:


doc_list, plot_list = preprocessor.read_csv()


# In[6]:


print(f'doc len : {len(doc_list)}')
print(f'plots len : {len(plot_list)}')


# In[7]:


norm_plot = preprocessor.stop_words(plot_list)
print(f'norm plots len : {len(norm_plot)}')


# In[8]:


# Tokenizing by keras
# texts_to_sequences
x_encoded, max_len, vocab_size = preprocessor.tokenizing(norm_plot)


# In[9]:


# max_len : 1649
print(x_encoded[:1])
print(max_len)


# In[10]:


#Padding
X_train = preprocessor.padding(x_encoded, maxlen=max_len, padding='post')


# In[11]:


#Glove
glove_embedding_matrix = preprocessor.embedding_glove(vocab_size)


# In[12]:


#Word2vec
word2vec_embedding_matrix = preprocessor.embedding_word2vec(vocab_size)


# In[ ]:


# x_train, glove_embedding_matrix, word2vec_embedding_matrix 
# embedding layer로


# In[ ]:




