#%%
# Load packages
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from tqdm import trange
import argparse
import time
import os

#%%
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--n_cls", type=int, default=8)
parser.add_argument("--beta", type=float, default=0.5)
opt = parser.parse_args()
# Hyper-parameters
n_clusters = opt.n_cls
beta = opt.beta
epsilon_EM = 1e-5
iter_max_EM = 40
iter_max_ICM = 10
root_path = "./imgs/mrf/{}".format(opt.name)
if not os.path.exists(root_path):
    os.makedirs(root_path)

#%%
# Utility functions
def label_to_img(labels):
    H, W = labels.shape
    colors = plt.cm.Set1
    img = [
        [
            int(255*colors(idx)[0]),
            int(255*colors(idx)[1]),
            int(255*colors(idx)[2]),
        ] 
        for idx in labels.reshape(-1)
    ]
    img = np.array(img).reshape(H, W, 3)
    return img

def initiate(x, n_clusters):
    """
    Args:
        x: (H, W, C)
        n_cluster: int
    Returns:
        x: (H, W)
        z0: (H, W)
        mu0: (K,)
        sigma0: (K,)
    """
    H, W, C = x.shape
    x = x.reshape(-1, C)
    # Kmeans clustering
    z0 = KMeans(n_clusters=n_clusters).fit_predict(x)
    x = np.mean(x, axis=-1)
    # Initialized mu0 and sigma0
    mu0 = np.zeros(n_clusters)
    sigma0 = np.zeros(n_clusters)
    for k in range(n_clusters):
        mu0[k] = np.mean(x[z0 == k])
        sigma0[k] = np.std(x[z0 == k])
    x = x.reshape(H, W)
    z0 = z0.reshape(H, W)

    # # naive-initiate
    # z0 = np.random.choice(list(range(n_clusters)), size=(H,W))
    # mu0 = np.random.rand(n_clusters) * 255
    # sigma0 = np.random.rand(n_clusters) * 5
    return x, z0, mu0, sigma0

def nlog_gaussian(x, mu, sigma):
    term1 = np.log(np.sqrt(2*np.pi)*sigma)
    term2 = (x - mu)**2 / (2 * sigma**2)
    return term1 + term2

def nlog_delta(all_z, z, i, j, beta):
    H, W = all_z.shape
    energy_list = []
    if i > 0:
        energy_list.append(int(all_z[i-1][j] != z) * 2 - 1)
    if i < H-1:
        energy_list.append(int(all_z[i+1][j] != z) * 2 - 1)
    if j > 0:
        energy_list.append(int(all_z[i][j-1] != z) * 2 - 1)
    if j < W-1:
        energy_list.append(int(all_z[i][j+1] != z) * 2 - 1)
    return 0.5 * beta * np.sum(energy_list)

def nlogp(x, z, mu, sigma, beta):
    """
    Compute the posterior.
    """
    energy_list = []
    H, W = x.shape
    print("Start to compute 'nlogp':")
    for i in trange(H):
        for j in range(W):
            xs = x[i][j]
            zs = z[i][j]
            # compute C1 energy
            e_c1 = nlog_gaussian(xs, mu[int(zs)], sigma[int(zs)])
            # compute c2 energy
            e_c2 = nlog_delta(z, zs, i, j, beta)
            energy_list.append(e_c1 + e_c2)
    energy = np.sum(energy_list)
    return energy

def icm(x, z, mu, sigma, beta, iter_max_ICM, n_clusters):
    """
    Descent algorithm.
    """
    H, W = x.shape
    for n in range(iter_max_ICM):
        print("Icm process: {}".format(n))
        new_z = np.zeros_like(z)
        for i in trange(H):
            for j in range(W):
                U_best = np.inf
                C_best = 0
                for k in range(n_clusters):
                    U = nlog_gaussian(x[i][j], mu[k], sigma[k]) \
                        + nlog_delta(z, k, i, j, beta)
                    if U < U_best:
                        U_best = U
                        C_best = k
                new_z[i][j] = C_best
        if np.sum(z != new_z) < 1:
            break
        z = new_z
    return z

def compute_prob(x, z, mu, sigma, beta, n_clusters):
    H, W = x.shape
    print("Compute prob-matrix:")
    prob_mat = np.zeros((H, W, n_clusters))
    for i in trange(H):
        for j in range(W):
            for k in range(n_clusters):
                term1 = nlog_gaussian(x[i][j], mu[k], sigma[k])
                term2 = nlog_delta(z, z[i][j], i, j, beta)
                p = np.exp(- term1 - term2)
                prob_mat[i][j][k] = p
    prob_mat = prob_mat / np.sum(prob_mat, axis=-1, keepdims=True)
    return prob_mat

#%%
# Read the image and initiate
img = Image.open("processed_image.png")
img = np.array(img)
H, W, C = img.shape
x, z, mu, sigma = initiate(img, n_clusters)

#%%
print("Initiate segmentation: ")
plt.imshow(label_to_img(z))
save_path = os.path.join(root_path, "init.png")
plt.savefig(save_path)

#%%
# EM algorithm
H, W = x.shape
for epoch in range(iter_max_EM):
    start = time.time()
    ### E-step
    print("=== E-step phase")
    # Compute previous U
    U_pre = nlogp(x, z, mu, sigma, beta)
    # Inference label using ICM
    z = icm(x, z, mu, sigma, beta, iter_max_ICM, n_clusters)
    # Compute current U
    U_cur = nlogp(x, z, mu, sigma, beta)
    # Judge whether to quit
    quit_index = abs(U_cur - U_pre) / abs(U_pre)
    print("Quit-index: {}".format(quit_index))
    if quit_index < epsilon_EM or np.isnan(quit_index):
        break

    ### M-step
    print("=== M-step phase")
    # Compute the prob-matrix
    prob_mat = compute_prob(x, z, mu, sigma, beta, n_clusters)
    # Update mu and sigma
    print("Update mu and sigma:")
    for k in trange(n_clusters):
        prob_slice = prob_mat[:, :, k]
        mu[k] = np.sum(x * prob_slice) / np.sum(prob_slice)
        sigma_sqr = np.sum((x - mu[k])**2 * prob_slice) / np.sum(prob_slice)
        sigma[k] = np.sqrt(sigma_sqr)
    end = time.time()
    print("Iter time: {}".format(end - start))
    
    # Show segmentation result
    plt.imshow(label_to_img(z))
    save_path = os.path.join(root_path, "{}.png".format(epoch))
    plt.savefig(save_path)
