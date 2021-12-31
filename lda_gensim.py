
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from PIL import Image
from gensim.models import LdaModel
from pandas.io import parsers
from tqdm import trange, tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_clusters", type=int, default=3)
opt = parser.parse_args()

#%%
exprData = pd.read_csv('Brain-expr_matrix.txt', index_col = 0, sep = '\t')
exprData = exprData.values
corpus = []
for j in trange(exprData.shape[1]):
    corpus.append([
        (i, exprData[i,j]) 
        for i in range(exprData.shape[0]) 
        if exprData[i,j] > 0
    ])

n_clusters = opt.n_clusters
start = time.time()
iterations = 2000
alpha = np.ones(n_clusters)
eta = np.ones(exprData.shape[0])

model = LdaModel(
    corpus=corpus,
    alpha=alpha,
    eta=eta,
    iterations=iterations,
    num_topics=n_clusters,
)

topics = model.get_document_topics(corpus)
topics = list(topics)
topics = [[prob[1] for prob in docum] for docum in topics]
topics = [np.argmax(prob) for prob in topics]
end = time.time()
print("Complete time: {}".format(end - start))

## visualization
spotPosition = pd.read_csv('Brain-spot_position.txt', sep = '\t')
position = [[item[1], item[2]] for item in spotPosition.values]

fig = plt.figure(figsize=(7, 8))
for item, cluster in tqdm(zip(position, topics)):
    plt.scatter(item[0], item[1], color=plt.cm.Set1(cluster+1), s=15)
plt.axis('off')
plt.savefig("./imgs/gensim/{}.png".format(n_clusters))
# plt.show()
