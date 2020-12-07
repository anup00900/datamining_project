import numpy as np
import pandas as pd
import math
import io
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pickle
from sentence_transformers import SentenceTransformer

# loading the pre-trained model
sbert_model= SentenceTransformer('bert-base-nli-mean-tokens')

# using pickle to save the model for using it later
filename = 'sbert_model.sav'
pickle.dump(sbert_model, open(filename, 'wb'))