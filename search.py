import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
data_file = 'Data/data_new.csv'
pickle_file = 'tfidf.pkl'
X_tfidf_file = 'X_tfidf.pkl'
data = pd.read_csv(data_file)
recommend = data[['Titles','key_notes']]
with open(pickle_file, 'rb') as file:
    vectorizer = pickle.load(file)
with open(X_tfidf_file, 'rb') as file:
    X_tfidf = pickle.load(file)


def recommend_movie(query):
    Y_tfidf = vectorizer.transform([query])
    sims = cosine_similarity(Y_tfidf, X_tfidf)
    movies = []
    for i in range(5):
        idx_sims_max = np.argmax(sims)
        movies.append(recommend['Titles'][idx_sims_max])
        sims[0][idx_sims_max] = -1
    return movies


def rec():
    try:
        i = 1
        while (i > 0):
            query = input("Enter The Name of a Movie or Tv Show: ")
            if query.lower() == 'quit':
                break
            else:
                print(recommend_movie(query.lower()))

    except KeyboardInterrupt:
        print("The movie or Tv Show does not exist\n")
        rec()

    except IndexError:
        print("The movie or Tv Show does not exist\n")
        rec()


print("To exit Enter \"quit\" \n")
rec()