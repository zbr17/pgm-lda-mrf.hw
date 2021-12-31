#%%
import numpy as np
from numpy.core.defchararray import lower
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
from tqdm import tqdm, trange
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="")
parser.add_argument("--multipler", type=int, default=1)
parser.add_argument("--rand", action="store_true", default=False)
opt = parser.parse_args()

#%%
## ----- 1.read data -----
exprData = pd.read_csv('Brain-expr_matrix-smallData.txt', index_col = 0, sep = '\t')
genes = exprData.index.tolist()


## ----- 2.convert to spot-gene (document-word) index list -----
gene_idx = dict(zip(genes, range(len(genes))))
exprDoc_idx = [[i for i in range(exprData.shape[0]) for k in range(exprData.iloc[i,j])] for j in range(exprData.shape[1])]


## ----- 3.initialize the hyper-parameters -----
K = 2
D = exprData.shape[1]
V = exprData.shape[0]
Ns = [len(item) for item in exprDoc_idx]
if not opt.rand:
    alpha = np.ones((1, K)) * opt.multipler
    eta = np.ones((1, V)) * opt.multipler
    name = "a{}b{}={}.png".format(opt.multipler, opt.multipler, opt.name)
else:
    alpha = np.random.rand(1, K) * opt.multipler
    eta = np.random.rand(1, V) * opt.multipler
    name = "rand-{}={}.png".format(opt.multipler, opt.name)
print(name)
iterations = 3000

#%%
## ----- 4.construct the LDA model and perform reference and learning
model = pm.Model()
with model:
    # Define the latent variables
    thetas = pm.Dirichlet("thetas", a=alpha, shape=(D, K))
    betas = pm.Dirichlet("betas", a=eta, shape=(K, V))
    zs = [
        pm.Categorical("z_d{}".format(d), p=thetas[d], shape=Ns[d])
        for d in range(D)
    ]
    # Define the observed variables
    ws = [
        pm.Categorical("w_{}_{}".format(d, i), p=betas[zs[d][i]], observed=exprDoc_idx[d][i])
        for d in range(D) for i in range(Ns[d])
    ]
print("Model constructed!")

#%%
# inference and learning
start = time.time()
with model:
    trace = pm.sample(iterations, progressbar=True)
end = time.time()
print("Model trained! Time cost: {}".format(end - start))

#%%
# ----- 5.assign the cluster label to each spot -----
split = 10
thetas_v = trace.get_values("thetas")
thetas_v = np.mean(thetas_v[(split-1)*len(thetas_v)//(split):], axis=0)
topics = np.argmax(thetas_v, axis=-1)

#%%
## ----- 6.visualize the cluster label on spatial position ------
spotPosition = pd.read_csv('Brain-spot_position-smallData.txt', sep = '\t')
position = [[item[1], item[2]] for item in spotPosition.values]
xlim = (np.min(spotPosition["x"])-1, np.max(spotPosition["x"])+1)
ylim = (np.min(spotPosition["y"])-1, np.max(spotPosition["y"])+1)

fig = plt.figure(figsize=(3, 5))
for item, cluster in tqdm(zip(position, topics)):
    plt.scatter(item[0], item[1], color=plt.cm.Set1(cluster+1), s=1000)
plt.axis('off')
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig("./imgs/pymc3/{}".format(name))

#%%
