import pandas as pd
import numpy as np
# from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt

import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, pearsonr, entropy
import libpysal
from esda.moran import Moran
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

import scipy.stats
from fastcluster import linkage
from scipy.spatial.distance import squareform, pdist
from matplotlib.patches import Ellipse
import scipy.linalg as linalg
import numpy.linalg as la
import seaborn as sns
from itertools import combinations

from skimage import measure

import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

import h5py

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Functions import *
import matplotlib.lines as mlines

# Set plot default
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams["axes.grid"] = False

import warnings
warnings.filterwarnings('ignore')


def box_plot(dictionary, var_name, value_name, save_title, box_width=0.8, xlabel_rot=0):

    # Make dataframe from dictionary
    df = pd.DataFrame(dictionary)

    # Melt the DataFrame to reshape it
    df = pd.melt(df, var_name=var_name, value_name=value_name)

    plt.figure(figsize=(4,3))
    sns.boxplot(data = df, x=var_name, y=value_name, boxprops=dict(alpha=.9),palette="muted",linewidth=0.7,fliersize=0.9, width=box_width)
    plt.ylabel(value_name,fontsize=10)
    plt.xlabel('')
    plt.xticks(rotation=xlabel_rot)
    plt.savefig(save_title+'.tiff', dpi=500, bbox_inches='tight')
    plt.show(block=False)
    return


def modified_raw_stress(embeddings1, embeddings2):
    """
    Compute raw stress between two sets of embeddings using Euclidean distance on a point by point basis

    Args:
        embeddings1 (numpy.ndarray): Array of embeddings from the first autoencoder.
        embeddings2 (numpy.ndarray): Array of embeddings from the second autoencoder.

    Returns:
        float: Stress value indicating dissimilarity between the embeddings.
    """
    # Ensure the embeddings have the same shape
    if embeddings1.shape != embeddings2.shape:
        raise ValueError("Embeddings must have the same shape")

    squared_diff = np.sum((embeddings1 - embeddings2)**2, axis=1)

    stress = np.mean(np.sqrt(squared_diff))
    return stress


def modified_norm_stress(embeddings1, embeddings2):
    """
    Compute normalized adjusted stress between two sets of embeddings on a structural basis

    Args:
        embeddings1 (numpy.ndarray): Array of embeddings from the first autoencoder.
        embeddings2 (numpy.ndarray): Array of embeddings from the second autoencoder.

    Returns:
        float: Adjusted MDS stress indicating dissimilarity between the embeddings.
    """
    # Ensure the embeddings have the same shape
    if embeddings1.shape != embeddings2.shape:
        raise ValueError("Embeddings must have the same shape")

    dists1 = squareform(pdist(embeddings1, 'euclidean'))
    dists2 = squareform(pdist(embeddings2, 'euclidean'))

    numerator = np.sum((dists1 - dists2) ** 2)
    denominator = np.sum(dists1 ** 2)

    if denominator == 0:
        return 0.0

    stress = np.sqrt(numerator / denominator)
    return stress