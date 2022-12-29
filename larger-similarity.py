#Imports
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import matplotlib.pyplot as plt
from scipy.stats import entropy
import urllib.request
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

#Function to preprocess the data (not much was needed here, but 
# might be more involved with a different dataset)
def preprocess(data):
    data["summary"] = data["description"]
    data.drop("subtitle", axis=1, inplace=True)
    data["title"] = [t.lower() for t in data["title"]]
    data.dropna(subset=['thumbnail', 'description', 'average_rating', 'num_pages'], inplace=True)
    data["authors"].fillna("no author on record", inplace=True)
    data["categories"].fillna("no category found", inplace=True)
    data.drop("isbn10", axis=1, inplace=True)
    data["published_year"] = [int(p) for p in data["published_year"]]

def clean_text(par):
    par = par.lower() #Changes everything to lowercase
    #Removes unwanted characters by replacing them with an empty string
    par = re.sub(r'\d+', '', par)
    par = re.sub(r'[^\w\s]', '', par)
    par = par.strip() #removes whitespaces
    par = nltk.word_tokenize(par) #tokenizes the text

    return par #returns the modified paragraph

def remove_stop_words(par):
    return [word for word in par if word not in stop_words] #Removes the stop words from NLTK. 

def stem_words(par):
    porter = PorterStemmer() 
    return [porter.stem(word) for word in par if len(word) > 1] #Uses porter stemmer to remove word stems (e.g. running - run)

#Helper function to apply the text preprocessing in order. 
def apply_all(par):
    return stem_words(remove_stop_words(clean_text(par)))

### LDA Topic Modeling
def train_lda(data):
    num_topics = 100 #Number of topics in the chunk
    chunksize = 500 #Chunk size
    dictionary = corpora.Dictionary(data["description"]) #Creates a dictionary object based on the book description
    corpus = [dictionary.doc2bow(doc) for doc in data['description']] #Creates corpus
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    #Uses gensim to create an LDA model based on the dictionary and corpus created above. 
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    return dictionary,corpus,lda

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_books(query,matrix,k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    most_sim_ids = sims.argsort()[:k]
    most_similar_df = df[df.index.isin(most_sim_ids)]
    return most_similar_df # the top k positional index of the smallest Jensen Shannon distances

## Plotting a histogram of the topic distribution for random book
def plotDoc(distribution):
    fig, ax = plt.subplots(figsize=(12,6));
    # the histogram of the data
    patches = ax.bar(np.arange(len(distribution)), distribution)
    ax.set_xlabel('Topic ID', fontsize=15)
    ax.set_ylabel('Topic Contribution', fontsize=15)
    ax.set_title("Topic Distribution for an Unseen Article", fontsize=20)
    ax.set_xticks(np.linspace(10,100,10))
    fig.tight_layout()
    plt.show()

#Similarity scoring
def get_similarity(lda, point, dictionary):
    new_bow = dictionary.doc2bow(df.iloc[point, 5])
    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)]) #tup[1] -> rating
    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
    return get_most_similar_books(new_doc_distribution, doc_topic_dist)

def display_image(link):
    #downloads the image from the provided link
    urllib.request.urlretrieve(link, "cover.png")
    im = Image.open("cover.png") #opens the downloaded image
#    im = im.resize((500,800)) #resizes it for better visibility
    im.show() #shows the image

#creating a new dataframe object
df = pd.read_csv("books.csv")
df.head()
preprocess(df)
df["description"] = df['description'].apply(apply_all)
#print(df["description"].head())

dictionary, corpus, lda = train_lda(df)
#trial = lda.show_topics(num_topics=10, num_words=20)
#print(trial)

name = ""
try:
    name = input("Enter a book name: ").lower()
except:
    print("Name not found in library")

print("You said: ", name)
title = []
for t in df["title"]:
    if name in t:
        title.append(t)

if len(title) > 1:
   print("There were multiple books matching you're query: ")
   [print(x) for x in title]
   j = int(input("Which book did yu want? ")) - 1
   print("You selected: ")
elif len(title) == 0:
    print("There are no titles matching your query. Please try again")
    exit()
else:
   j = 0
print(title[j])

indx = int(df["title"].loc[lambda x: x==title[j]].index[0])

res = get_similarity(lda, indx, dictionary)
name = res["title"]
img = res["thumbnail"]

for item, url in zip(name, img):
    print(item)
    display_image(url)
