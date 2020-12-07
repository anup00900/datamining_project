import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
import math
import nltk

sample_data= pd.read_csv('prepared_data.csv')
sample_duplicates= pd.read_csv('prepared_duplicates.csv')
sample_limit= len(sample_data)
# Taking the first sample_limit number of sample questions
sample_data= sample_data[:sample_limit]

# loading the pre-trained use_model
path= 'https://tfhub.dev/google/universal-sentence-encoder/4'
use_model= hub.load(path)

# importing the use_embedding
with open('use_embedding.data','rb') as filehandle:
    use_embedding= pickle.load(filehandle)

def cosine(u, v):
    ''' this function returns the cosine similarity b/w two vectors'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# finding the similarity scores for each query with all the questions in corpus- We have used three similarity measures
use_cosine_simscores=[]

test_questions= list(sample_duplicates['preparedtitle'])

for i in range(len(test_questions)):
    query= test_questions[i]
    query_vec = use_model([query])[0]

    cosine_simscore=[] # list of similarity scores with all the questions for this current query.

    for s in use_embedding:
        cosine_sim= cosine(query_vec, s)

        cosine_simscore.append(cosine_sim)

    # now we have the similarity scores with all the questions for this current query using all the three similarity measures.
    use_cosine_simscores.append(cosine_simscore)

use_cosine_simscores= np.array(use_cosine_simscores)

# saving the use_cosine_simscores
with open('use_cosine_simscores.data', 'wb') as filehandle:
    pickle.dump(use_cosine_simscores, filehandle)
