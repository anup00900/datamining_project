import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import demoji
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
demoji.download_codes()


def preProcessing(text):
    text=text.lower()

    lemmatizer=WordNetLemmatizer()
    words=text.split(' ')
    
    stopSet = stopwords.words('english')
    text = " ".join([i for i in words if i not in stopSet])

    #replace emoji
    words=text.split(' ')
    idx=0;
    for word in words:
        emojiDict=demoji.findall(word);
        emojiText=list(emojiDict.keys());
        if len(emojiText)>0:
            words[idx]=emojiDict[emojiText[0]];
        idx+=1;
    
    text=''
    for w in words:
        nWord=w.replace('n\'t','not')
        text+=' '+lemmatizer.lemmatize(nWord)

    
    
    #remove non-ascii characters
    text = unidecode(text)

    seperators=list(string.punctuation)
    for i in seperators:
        text=text.replace(i,' ')
    
    #print(text)
    return text
