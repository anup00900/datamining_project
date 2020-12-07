import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle
import pandas as pd

# loading the pre-trained model
path= 'https://tfhub.dev/google/universal-sentence-encoder/4'
use_model= hub.load(path)

#--------------------- Instead of saving the model(giving me some problems) we will try to encode the questions here and then directly save the embeddings------#
#So essentially use_model and use_encode are combined
# importing the train set
sample_data= pd.read_csv('prepared_data.csv')
sample_limit= len(sample_data)

# Taking the first sample_limit number of sample questions
sample_data= sample_data[:sample_limit]
questions= list(sample_data['preparedtitle'])

# encoding the questions
use_embedding= use_model(questions)

# saving the embeddings so that query processing becomes easy
with open('use_embedding.data', 'wb') as filehandle:
    pickle.dump(use_embedding, filehandle)
