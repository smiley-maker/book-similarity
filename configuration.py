import pandas as pd

#Reading in the two datasets
info = pd.read_csv("collaborative_book_metadata.csv", index_col=0)
ratings = pd.read_csv("collaborative_books_df.csv", index_col=0)

#Now we have two datasets with different information. 
#The 'info' dataframe includes the book id, page count,
#description, etc. and the ratings data includes numerous 
#ratings given by different users for each book. 

#I wanted to get an average rating for each book rather than
#having specific user ratings. 
newRatings = []
titles = set(ratings["title"]) #Gives unique titles

#Looping through all of the unique titles
for title in titles:
    r = round(sum(ratings[ratings["title"] == title]["Actual Rating"]/len(ratings[ratings["title"] == title])), 2)
    #Sums over all reviews for a given book and divides 
    #by the total number of reviews and rounded. 
    newRatings.append(r) #Appends the average user rating

#Creates a dictionary of the results I wanted
#from the ratings data. 
results = {
    "title": list(titles),
    "rating": newRatings
}

results = pd.DataFrame(results) #Changes the dictionary to a dataframe
#This merges the two dataframes based on the book title. 
df = pd.merge(info, results, on="title")
#I decided to save the modified data to a new csv file
#so that I could work with it a little easier in
#new Python scripts. 
df.to_csv("data.csv")