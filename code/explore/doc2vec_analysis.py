#!/usr/bin/env python3

from collections import Counter
from gensim.models import Doc2Vec, Phrases
from gensim.models.doc2vec import LabeledSentence
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np 
import pandas as pd
from pprint import pprint
from random import shuffle
import re
import string

letters = set(string.ascii_lowercase)
numbers = set(string.digits)
stop_words = STOPWORDS.union(letters).union(numbers)


def preprocess(text):
    """Remove punctuation and convert words to lower case"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower().split()

def train_bigrammer(df, text_col):
    """Detect two-word phrases, e.g., heavy_metal"""
    return Phrases(map(preprocess, df[text_col].tolist())) 

def create_labeled_sentences(text, title, drop_stopwords=True):
    """Generate doc2vec LabeledSentence objects from documents. """
    doc_tag = '_'.join(preprocess(title))    
    doc_words = list(filter(lambda word: word not in stop_words,
                      bigrams[preprocess(text)]))
    return LabeledSentence(doc_words, [doc_tag])

def create_docs(df, id_col, text_col):
    return [create_labeled_sentences(text, title) for text, title in
    zip(df[text_col].tolist(), df[id_col].tolist())]

# ----------------------------------
# Preprocess text for doc2vec model
# ----------------------------------

file = "reviews.csv"
df = pd.read_csv(file, header=None,
    names=['band','album','rating','votes','review'],
    dtype={'band':str,'album':str,'rating':np.float64,'review':str})
df = df.dropna()

#df = df.head(100)
df['band_code'] = df['band'].astype('category').cat.codes
df['band_album'] = df['band'] +': ' + df['album']

bigrams = train_bigrammer(df, "review")

docs = create_docs(df, "band", "review")
shuffle(docs)

# ----------------------------------
# Train the model
# ----------------------------------
model = Doc2Vec(dm=1, dbow_words=1, min_count=4, negative=3,
                hs=0, sample=1e-4, window=10, size=100, workers=8)

model.build_vocab(docs)

from gensim.models.word2vec import Word2Vec
model.load_word2vec_format('/home/edward/work/projects/finance/data/GoogleNews-vectors-negative300.bin', binary=True)
model.train(docs)


# model.save('{}/model_objects/model.doc2vec'.format(project_dir))
# model = Doc2Vec.load('{}/model_objects/model.doc2vec'.format(project_dir))

# ----------------------------------
# Using the model
# ----------------------------------

# Find words similar to query word
pprint(model.docvecs.most_similar(positive= ['agalloch']))


pprint(model.most_similar(positive=['agalloch']))

# Find bands similar to query word
vec = model['tool']
pprint(model.docvecs.most_similar([vec]))

# find bands similar to query band
pprint(model.docvecs.most_similar(positive=['the_mantle', 'ghost_reveries']))#, negative=['heritage']))


# find bands similar to query band
for i in df.band.tolist()[:1]:
    print("\nQuery: {}".format(i))
    doctag = '_'.join(preprocess(i))
    pprint(model.docvecs.most_similar(doctag))

# Find words similar to a query band
for i in df.band.tolist()[:1]:
    print("\nQuery: {}".format(i))
    doctag = '_'.join(preprocess(i))
    vec = model.docvecs[doctag]
    pprint(model.most_similar([vec]))


# ----------------------------------
# Visualize the embedding space the model
# ----------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import manifold

 
def plot_embedding(id,id_2, method="TSNE", n_comps=30, perplexity=10, save_as=''):
    doctags = ['_'.join(preprocess(i)) for i in id]
    vecs = np.array([unitvec(model.docvecs[doctag]) for doctag in doctags])    
    # First doing pca on the vectors can reduce the noise and yield a better
    # 2d projection
    pca_vecs = PCA(n_components=n_comps).fit(vecs).transform(vecs)
    if method is "TSNE":
        embedding = manifold.TSNE(n_components=2, random_state=0, perplexity=perplexity)
        model_vecs = embedding.fit_transform(pca_vecs)
    
    elif method is "SpectralEmbedding":
        embedding = manifold.SpectralEmbedding(n_components=2, random_state=0,
            eigen_solver="arpack")
        model_vecs = embedding.fit_transform(pca_vecs)
    else:
        print("Invalid embedding method. Use either 'TSNE' or 'SpectralEmbedding'")

    fig, ax = plt.subplots(figsize=(24, 24))
    ax.scatter(model_vecs[:,0],model_vecs[:,1], c=np.array(df.query(query)['band_code'][:top_n]),
         edgecolors='black', s=100, cmap='viridis')

    # Annotate points with the id
    for i, title in enumerate(id_2):
        ax.annotate(title, 
                    xy=(model_vecs[i,0], model_vecs[i,1]), 
                    fontsize=12, alpha=.9)

    if save_as:
        fig.savefig(save_as, dpi=fig.dpi)
    return fig

# Plot top bands
top_n = 100

query = 'rating > 8 and votes > 1000'
titles = df.query(query).band.tolist()[:top_n]
band_album = df.query(query).album.tolist()[:top_n]

fig = plot_embedding(titles, band_album, method="TSNE", save_as='{}/plots/top_metal_bands.png'.format(project_dir))
fig.show()


# Plot top albums
query = 'rating > 8 and votes > 1000'
titles = df.query(query).album.tolist()[:top_n]
fig = plot_embedding(titles, method="SpectralEmbedding", save_as='{}/plots/top_metal_albums.png'.format(project_dir))
fig.show()


from mpl_toolkits.mplot3d import Axes3D
def plot_3d_embedding(id, method="TSNE", n_comps=30, perplexity=10, save_as=''):
    doctags = ['_'.join(preprocess(i)) for i in id]
    vecs = np.array([unitvec(model.docvecs[doctag]) for doctag in doctags])    
    # First doing pca on the vectors can reduce the noise and yield a better
    # 2d projection
    pca_vecs = PCA(n_components=n_comps).fit(vecs).transform(vecs)
    if method is "TSNE":
        embedding = manifold.TSNE(n_components=3, random_state=0, perplexity=perplexity)
        model_vecs = embedding.fit_transform(pca_vecs)
    
    elif method is "SpectralEmbedding":
        embedding = manifold.SpectralEmbedding(n_components=3, random_state=0,
            eigen_solver="arpack")
        model_vecs = embedding.fit_transform(pca_vecs)
    else:
        print("Invalid embedding method. Use either 'TSNE' or 'SpectralEmbedding'")

    fig = plt.figure(figsize=(24, 24))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model_vecs[:,0],model_vecs[:,1],model_vecs[:,2], c=np.array(df.query(query)['band_code'][:top_n]),
         edgecolors='black', cmap='viridis')

    if save_as:
        fig.savefig(save_as, dpi=fig.dpi)
    return fig
