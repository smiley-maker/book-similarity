#Imports
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import numpy as np
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import matplotlib.pyplot as plt
from scipy.stats import entropy
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import warnings
warnings.filterwarnings("ignore")

#Function to preprocess the data (not much was needed here, but 
# might be more involved with a different dataset)
def preprocess(data):
    #I decided to create a duplicate of the summary column
    #so that I could view the original summaries again later
    data["summary"] = data["description"]
    data.drop("book_id_mapping", inplace=True, axis=1) #Not needed for my application

def clean_text(par):
    par = par.lower() #Changes everything to lowercase
    #Removes unwanted characters by replacing them with an empty string
    par = re.sub(r'\d+', '', par)
    par = re.sub(r'[^\w\s]', '', par)
    par = par.strip() #removes whitespaces
    par = nltk.word_tokenize(par) #tokenizes the text

    return par #returns the modified paragraph

def remove_stop_words(par):
    return [word for word in par if word not in stop_words]

def stem_words(par):
    porter = PorterStemmer()
    return [porter.stem(word) for word in par if len(word) > 1]

def apply_all(par):
    return stem_words(remove_stop_words(clean_text(par)))

#creating a new dataframe object
df = pd.read_csv("data.csv", index_col=0)
preprocess(df)
df["description"] = df['description'].apply(apply_all)
#print(df["description"].head())