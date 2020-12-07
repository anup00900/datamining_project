import numpy as np
import pandas as pd
import math
import io
import nltk
from nltk.tokenize import word_tokenize
import pickle
from sentence_transformers import SentenceTransformer

sample_data= pd.read_csv('prepared_data.csv')
sample_limit= len(sample_data)

# Taking the first sample_limit number of sample questions
sample_data= sample_data[:sample_limit]
questions= list(sample_data['preparedtitle'])

# importing the sbert model
filename= 'sbert_model.sav'
sbert_model= pickle.load(open(filename, 'rb'))

# encoding the questions
sbert_embedding= sbert_model.encode(questions)

# saving the embeddings so that query processing becomes easy
with open('sbert_embedding.data', 'wb') as filehandle:
    pickle.dump(sbert_embedding, filehandle)