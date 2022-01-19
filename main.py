import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob

if __name__ == '__main__':

    # Reading text from file
    text = open('a3.txt')
    text = text.read()
    blob = TextBlob(text)

    # Deleting stopwatches from text
    sens = []
    for iy in range(0, len(blob.sentences) - 1):
        sens.append(remove_stopwords(blob.sentences[iy].string))

    # Number of sentences on text
    length = len(sens)

    # Bert model
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(sens)

    # Creating array for similarity matrix
    mat = np.zeros((length, length))

    # Counting cosine similarity for each sentence, adding it to similarity matrix
    for ix in range(0, length):
        for iy in range(0, length):
            if ix != iy:
                mat[ix, iy] = cosine_similarity([embeddings[ix]], embeddings[iy:iy + 1])

    # Sum of similarities of each sentence
    summ = np.zeros(length)
    for ix in range(0, length):
        for iy in range(0, length):
            summ[ix] += mat[iy, ix]

    # Sorting array of sums to obtain the biggest values
    out = np.array(sorted(((value, index) for index, value in enumerate(summ)), reverse=True))

    # Picking number of sentences with the biggest similarity values
    out = out[0:5]
    out = sorted(out, key=lambda x: x[1])

    # Showing full sentences (with stopwords) that were picked as summary
    for ix in range(0, 5):
        print(blob.sentences[int(out[ix][1])].string)
