import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


# Exception for too big number of summary
class TooBigSummarySize(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# Reading text from file
def read_text(filename):
    text = open(filename)
    text = text.read()
    blob = TextBlob(text)
    return blob


# Deleting stopwatches from text
def delete_stopwords(blob):
    sentences = []
    for x in range(0, len(blob_text.sentences) - 1):
        sentences.append(remove_stopwords(blob_text.sentences[x].string))
    return sentences


# Counting cosine similarity for each sentence, adding it to similarity matrix
def sentence_similarity(embeds, length):
    mat = np.zeros((length, length))
    for ix in range(0, length):
        for iy in range(0, length):
            if ix != iy:
                mat[ix, iy] = cosine_similarity([embeds[ix]], embeds[iy:iy + 1])
    return mat


# Sum of similarities of each sentence
def similarity_sum(mat, length):
    sim_sum = np.zeros(length)
    for ix in range(0, length):
        for iy in range(0, length):
            sim_sum[ix] += mat[iy, ix]
    return sim_sum


# Sorting array of sums to obtain the biggest values
# Picking number of sentences with the biggest similarity values
def sort_and_choose_sentences(sim_sum, size):
    out = np.array(sorted(((value, index) for index, value in enumerate(sim_sum)), reverse=True))
    out = out[0:size]
    out = sorted(out, key=lambda x: x[1])
    return out


if __name__ == '__main__':
    try:
        # Deciding size of summary we want to create
        summary_size = 33

        # Reading text from file
        blob_text = read_text('file.txt')

        # Deleting stopwatches from text
        sentences = delete_stopwords(blob_text)

        # Number of sentences on text
        length = len(sentences)
        if summary_size > length:
            raise TooBigSummarySize("Choose smaller number of summary size, thanks!")

        # Bert model
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings = model.encode(sentences)

        # Counting cosine similarity for each sentence, adding it to similarity matrix
        matrix = sentence_similarity(embeddings, length)

        # Sum of similarities of each sentence
        sim_sum = similarity_sum(matrix, length)

        # Sorting array of sums to obtain the biggest values
        # Picking number of sentences with the biggest similarity values
        sorted_sentences = sort_and_choose_sentences(sim_sum, summary_size)

        # Showing full sentences (with stopwords) that were picked as summary
        for ix in range(0, summary_size):
            print(blob_text.sentences[int(sorted_sentences[ix][1])].string)

    except TooBigSummarySize as error:
        print("Attention!!! You created a new error:", error.value)
