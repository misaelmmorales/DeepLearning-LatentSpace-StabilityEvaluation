# import matplotlib as mpl
# from sklearn.cluster import KMeans
# from sklearn.datasets import load_wine
# from sklearn.preprocessing import StandardScaler
# from scipy.optimize import linear_sum_assignment
# from scipy.stats import spearmanr, pearsonr, entropy
# import libpysal
# from esda.moran import Moran
# from scipy.spatial import ConvexHull
# from shapely.geometry import Polygon
# from itertools import combinations
# import h5py
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset
# from matplotlib.colors import ListedColormap
# from matplotlib.lines import Line2D
# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load Packages
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.stats
from scipy.stats import gaussian_kde
import scipy.linalg as linalg
from scipy.spatial.distance import squareform, pdist
from fastcluster import linkage

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import seaborn as sns

import torch
import torch.linalg
from skimage import measure

# Set plot default
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams["axes.grid"] = False

import warnings
warnings.filterwarnings('ignore')


def reset_environment(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()


def mvee(points, tol=0.0001):
    """
    Function to compute the minimum volume enclosing ellipsoid (MVEE) adapted and modified from
    https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953

    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol +1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx ] - d -1.0 ) /(( d +1 ) *(M[jdx ] -1.0))
        new_u = ( 1 -step_size ) *u
        new_u[jdx] += step_size
        err = la.norm(new_u -u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c) ) /d
    return A, c


def mvee_parallel(points, tol=0.0001):
    """
    Function to compute the minimum volume enclosing ellipsoid (MVEE) adapted from
    https://gist.github.com/jasnyder/ccdf5ca4e76e81a2047c78887a95e0a2

    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """

    device = torch.device('cuda')
    dot = lambda foo, bar : torch.tensordot(foo, bar, dims=1)
    N, d = points.shape
    Q = torch.column_stack((points, torch.ones(N, device=device))).T # Q.shape = (d+1, N)
    err = tol + 1.0
    u = torch.ones(N, device = device)/N # u.shape = (N,)
    while err > tol:
        X = dot(Q * u[None, :], Q.T) # shapes: (((d+1, N), (N, N)) , (N, d+1)) = (d+1, d+1)
        M = torch.sum(dot(Q.T, torch.linalg.inv(X) ) *Q.T, dim = 1) # (((N, d+1), (d+1, d+1)), (d+1, N)) = (N, N);
                                                                    # M.shape = (N,)
        jdx = torch.argmax(M)
        step_size = (M[jdx ] - d -1.0 ) /(( d +1 ) *(M[jdx ] -1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = torch.linalg.norm(new_u-u)
        u = new_u
    c = dot(u, points)
    A = torch.linalg.inv(dot(points.T * u[None, :], points)
                         - torch.outer(c, c) ) /d
    return A, c


def dist_2_cent(x, y, center):
    '''
    Obtain distance to center coordinates for the entire x,y array passed.
    '''

    # delta_x, delta_y = abs(x - center[0]), abs(y - center[1])
    delta_x, delta_y = (x - center[0]), (y - center[1])
    dist = np.sqrt(delta_x ** 2 + delta_y ** 2)

    return delta_x, delta_y, dist


def get_outer_shell(center, x, y):
    '''
    Selects those stars located in an 'outer shell' of the points cloud,
    according to a given accuracy (ie: the 'delta_angle' of the slices the
    circle is divided in).
    '''

    delta_x, delta_y, dist = dist_2_cent(x, y, center)

    # Obtain correct angle with positive x axis for each point.
    angles = []
    for dx, dy in zip(*[delta_x, delta_y]):
        ang = np.rad2deg(np.arctan(abs(dx / dy)))
        if dx > 0. and dy > 0.:
            angles.append(ang)
        elif dx < 0. and dy > 0.:
            angles.append(180. - ang)
        elif dx < 0. and dy < 0.:
            angles.append(270. - ang)
        elif dx > 0. and dy < 0.:
            angles.append(360. - ang)

    # Get indexes of angles from min to max value.
    min_max_ind = np.argsort(angles)

    # Determine sliced circumference. 'delta_angle' sets the number of slices.
    delta_angle = 1.
    circle_slices = np.arange(delta_angle, 361., delta_angle)

    # Fill outer shell with as many empty lists as slices.
    outer_shell = [[] for _ in range(len(circle_slices))]
    # Initialize first angle value (0\degrees) and index of stars in list
    # ordered from min to max distance value to center.
    ang_slice_prev, j = 0., 0
    # For each slice.
    for k, ang_slice in enumerate(circle_slices):
        # Initialize previous maximum distance and counter of stars that have
        # been processed 'p'.
        dist_old, p = 0., 0
        # For each star in the list, except those already processed (ie: with
        # an angle smaller than 'ang_slice_prev')
        for i in min_max_ind[j:]:
            # If the angle is within the slice.
            if ang_slice_prev <= angles[i] < ang_slice:
                # Increase the index that stores the number of stars processed.
                p += 1
                # If the distance to the center is greater than the previous
                # one found (if any).
                if dist[i] > dist_old:
                    # Store coordinates of new star farthest away from center
                    # in this slice.
                    outer_shell[k] = [x[i], y[i]]
                    # Re-assign previous max distance value.
                    dist_old = dist[i]
            # If the angle value is greater than the max slice value.
            elif angles[i] >= ang_slice:
                # Increase index of last star processed and break out of
                # stars loop.
                j += p
                break

        # Re-assign minimum slice angle value.
        ang_slice_prev = ang_slice

    # Remove empty lists from array (ie: slices with no stars in it).
    outer_shell = np.asarray([x for x in outer_shell if x != []])

    return outer_shell


def run_mvee(array_2d, plotter=True, parallelize=False):
    # center the 2d array at 0.0 i.e., mean centering
    data_array = array_2d - np.mean(array_2d, axis=0)

    # Extract x and y coordinates from mean centered array
    x, y = data_array[:, 0], data_array[:, 1]

    # Use the centroid of data as the initial center for get_outer_shell
    center = [np.mean(x), np.mean(y)]
    points = get_outer_shell(center, x, y)

    if not parallelize:
        # Run the MVEE algorithm based on numpy
        A, centroid = mvee(points)
    else:
        A, centroid = mvee_parallel(points)

    # Extract ellipse parameters
    U, D, V = la.svd(A)
    rx, ry = 1./np.sqrt(D)
    dx, dy = 2 * rx, 2 * ry
    a, b = max(dx, dy), min(dx, dy)
    mvee_anis = a/ b
    alpha = np.rad2deg(np.arccos(V[0][1]))

    if plotter:
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=10, zorder=4)
        ax.scatter(points[:, 0], points[:, 1], s=75, c='r', zorder=3)
        ax.scatter(*centroid, s=70, c='g')
        ax.scatter(center[0], center[1], c='k', s=70)
        ellipse = Ellipse(xy=centroid, width=a, height=b, edgecolor='k',
                          angle=alpha, fc='None', lw=2)
        plt.savefig('MVEE Anisotropy Plot.tiff', dpi=300, bbox_inches='tight')
        ax.add_patch(ellipse)
        plt.show(block=False)
    return mvee_anis


def compute_anisotropy(array_2d, type, plotter=True):

    # Initialize variables
    global_anis = None
    local_anis = []

    # Perform mean centering to 0 on array
    temp = array_2d.copy()
    data = temp - np.mean(temp, axis=0)
    center = np.mean(data, axis=0)

    # Estimate the PDF using Gaussian Kernel Density Estimation
    kde = gaussian_kde(data.T)
    x_grid, y_grid = np.mgrid[data[:,0].min():data[:,0].max():100j, data[:,1].min():data[:,1].max():100j]
    pdf_values = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))

    # Reshape for contour plot
    pdf_values = pdf_values.reshape(x_grid.shape)

    # Make the plot
    if plotter:
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(4, 4)

        # Main scatter plot
        ax_main = plt.subplot(gs[1:4, 0:3])

        if type.lower() == 'local':
            level_95 = np.percentile(pdf_values, 95)
            levels_below_95 = np.linspace(pdf_values.min(), level_95, num=5, endpoint=False)
            linewidths = np.linspace(0.5, 1.5, num=len(levels_below_95)) # make linewidths increasing wrt level
            contour_lines = ax_main.contour(x_grid, y_grid, pdf_values, levels=levels_below_95, colors='black',
                                            linewidths=linewidths)
            contour_lines_95 = ax_main.contour(x_grid, y_grid, pdf_values, levels=[level_95], colors='blue',
                                               linestyles='dashed', linewidths=1.75)
            ax_main.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
            ax_main.clabel(contour_lines_95, inline=True, fontsize=8, fmt='%.2f')

        else:
            levels = np.linspace(pdf_values.min(), pdf_values.max(), num=6, endpoint=False)
            linewidths = np.linspace(0.5, 1.75, num=len(levels))
            contour_lines_95 = ax_main.contour(x_grid, y_grid, pdf_values, levels=levels, colors='black',
                                               linewidths=linewidths) # for entire pdf, not at 95 percentile
                                # like the name suggests. This is because plotting objects cannot be copied
            ax_main.clabel(contour_lines_95, inline=True, fontsize=8, fmt='%.2f')
            ax_main.plot(data[:, 0], data[:, 1], 'k.', markersize=2)

        ax_main.scatter(center[0], center[1], c='r', s=2, zorder=3)
        ax_main.set_xlabel('LS 1')
        ax_main.set_ylabel('LS 2')

        # Marginal distributions
        ax_xDist = plt.subplot(gs[0, 0:3], sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)

        # Plotting marginal distributions as KDE's
        kde_x = gaussian_kde(data[:, 0])
        x_line = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000)
        ax_xDist.plot(x_line, kde_x(x_line) * np.diff(x_line).mean() * len(data[:, 0]), color='black')
        ax_xDist.set_ylabel('Density')

        kde_y = gaussian_kde(data[:, 1])
        y_line = np.linspace(data[:, 1].min(), data[:, 1].max(), 1000)
        ax_yDist.plot(kde_y(y_line) * np.diff(y_line).mean() * len(data[:, 1]), y_line, color='black')
        ax_yDist.set_xlabel('Density')

        # Turn off ticks for marginal distributions
        ax_xDist.xaxis.set_tick_params(labelbottom=False)
        ax_yDist.yaxis.set_tick_params(labelleft=False)
        plt.subplots_adjust(wspace=0, hspace=0)

    if type.lower() == 'global':
        # Compute covariance matrix for the entire dataset
        cov = np.cov(data.T)
        eigenvalues, eigenvectors = linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # Ellipse parameters for the entire dataset
        width, height = 2 * np.sqrt(eigenvalues)
        overall_ellipse = Ellipse(xy=np.mean(data, axis=0), width=width, height=height, angle=angle, edgecolor='red',
                                  fc='None', lw=2)

        # Compute anisotropy global
        global_anis = width / height

    elif type.lower() == 'local':
        # Contour level at 95% confidence interval
        level = np.percentile(pdf_values, 95)

        if plotter:
            ax_main.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
            # Find and fit ellipses to all such contours, which is indicative of tentative cluster structures within
            # an array if any
            for path in contour_lines_95.collections[0].get_paths(): # Grab paths of 95% CI
                contains_points = path.contains_points(data)
                ax_main.plot(data[contains_points, 0], data[contains_points, 1], 'b.', markersize=2) # Plot data points
                # inside the 95% CI
                v = path.vertices
                cov = np.cov(v, rowvar=False)
                eigenvalues, eigenvectors = linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                angle = np.arctan2(*eigenvectors[:, 0][::-1])
                width, height = 2 * np.sqrt(eigenvalues)
                ellipse = Ellipse(xy=np.mean(v, axis=0), width=width, height=height, angle=np.degrees(angle),
                                  edgecolor='red', fc='None')

                # Add each ellipse to the plot
                if plotter:
                    #ax.add_patch(ellipse)
                    ax_main.add_patch(ellipse)

                # Compute local anisotropies with respect to bi or multimodal pdf's and save
                local_anis.append(width / height)
        else:
            # Find and fit ellipses to all such contours, which is indicative of tentative cluster structures within an
            # array if any
            contours = measure.find_contours(pdf_values, level=level)
            for contour in contours:
                v = contour
                cov = np.cov(v, rowvar=False)
                eigenvalues, eigenvectors = linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                angle = np.arctan2(*eigenvectors[:, 0][::-1])
                width, height = 2 * np.sqrt(eigenvalues)
                ellipse = Ellipse(xy=np.mean(v, axis=0), width=width, height=height, angle=np.degrees(angle),
                                  edgecolor='red', fc='None')

                # Compute local anisotropies with respect to bi or multimodal pdf's and save
                local_anis.append(width / height)

    # Compute harmonic mean of the local anisotropy
    harmonic_anis = scipy.stats.hmean(local_anis) if local_anis else None

    if  plotter and type.lower() == 'global':
        ax_main.add_patch(overall_ellipse)
        black_line = mlines.Line2D([], [], color='black', label='PDF')
        red_line = mlines.Line2D([], [], color='red', label='Global Anisotropy')
        data_pt = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=1, label='Sample')

        # Add the legend to the plot
        ax_main.legend(handles=[data_pt, black_line, red_line], loc='best')
        plt.savefig('Global Anisotropy.tiff', dpi=300, bbox_inches='tight')
        plt.show(block=False)

    if plotter and type.lower() == 'local':
        # Create proxy artists for the legend
        black_line = mlines.Line2D([], [], color='black', label='PDF')
        blue_line = mlines.Line2D([], [], color='blue', linestyle='dashed', label='95% CI')
        red_line = mlines.Line2D([], [], color='red', label='Local Anisotropy')
        data_pt = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=1, label='Sample')

        # Add the legend to the plot
        ax_main.legend(handles=[data_pt, black_line, blue_line, red_line], loc='best')
        plt.savefig('Local Anisotropy.tiff', dpi=300, bbox_inches='tight')
        plt.show(block=False)
    return global_anis, local_anis, harmonic_anis


def percentage_change(measure, data_type):

    """
    Finds the percentage change in any measure of choice for each consecutive pair of AE realizations i.e., in a
    sequential manner, thus, not a full combinatorial computation.
    :param realizations:
    :return:
    """
    changes = []

    if data_type.lower() == 'list':
        for i in range(0, len(measure)):
            prev_set = set(measure[i - 1])
            current_set = set(measure[i])
            changed_elements = current_set.symmetric_difference(prev_set)
            total_elements = len(prev_set.union(current_set))
            change = len(changed_elements) / total_elements * 100
            changes.append(change)

    elif data_type.lower() == 'numpy':
        for i in range(0, len(measure)):
            prev_row = measure[i - 1]
            current_row = measure[i]
            change = np.abs((current_row - prev_row) / prev_row) * 100
            changes.append(change)
    return changes


def adjusted_stress(embeddings1, embeddings2):
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
    denominator = np.sum(dists1 * dists2)

    if denominator == 0:
        return 0.0

    stress = np.sqrt(numerator / denominator)
    return stress


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    This function checks if the distance matrix is symmetric, prior to making a sorted dissimilarity matrix
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def seriation(Z, N, cur_index):
    """
    This is a function that creates a sorted 2D matrix as a figure

        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    """

    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """
        input:
            - dist_mat is a distance matrix
            -  = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarchical tree
            - res_linkage is the hierarchical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    """

    N = len(dist_mat)
    flat_dist_mat = dist_mat if len(dist_mat.shape) == 2 else squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def weighted_percentile(data, weights, perc): # finds weighted percentiles
    if len(data) != len(weights):
        raise ValueError("Data and weights must be the same length")

    # Assert non-negative weights
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

    # Sort dta and weigh
    ix = np.argsort(data)
    data_sorted = np.array(data)[ix]
    weights_sorted = np.array(weights)[ix]
    # Find CDF of weights
    cdf = np.cumsum(weights_sorted) / np.sum(weights_sorted)
    # Bin centering for cdf to 0
    cdf = np.insert(cdf, 0, 0)
    cdf = (cdf[:-1] + cdf[1:]) / 2
    return np.interp(perc, cdf, data_sorted)


def knuth_bin_width(data):
    """
    Calculate the optimal number of bins using Knuth's Rule.

    Parameters:
    data (array-like): The input data for which you want to create a histogram.

    Returns:
    int: The recommended number of bins.
    """
    N_data = len(data)
    if N_data <= 0:
        raise ValueError("Data array must have at least one element")

    # Calculate the number of bins using Knuth's Rule formula
    optimal_bin = 1 + np.log2(N_data) + np.log2(1 + np.sqrt(N_data))

    # Round N to the nearest integer
    optimal_bin = int(np.round(optimal_bin))
    return optimal_bin


def histogram_bounds(ax_or_plt, optimal_bin, values, weights, color, max_freq_override=None): # finds uncertainty bounds p10, p50, p90
    hist_data, bin_edges = np.histogram(values, bins=optimal_bin)
    max_freq = max(hist_data)
    max_freq += max_freq * 0.05

    if max_freq_override is not None:
        max_freq = max_freq_override

    p10 = weighted_percentile(values, weights, 0.1)
    p50 = np.average(values, weights=weights)
    p90 = weighted_percentile(values, weights, 0.9)
    plot_function = ax_or_plt if hasattr(ax_or_plt, 'plot') else plt

    plot_function.plot([p10, p10], [0.0, max_freq], color=color, linestyle='dashed', label='P10')
    plot_function.plot([p50, p50], [0.0, max_freq], color=color, label='P50')
    plot_function.plot([p90, p90], [0.0, max_freq], color=color, linestyle='dotted', label='P90')


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


def make_boxplot(dictionary, y_label, c_str, box_width=0.3): # 0.6 is a fat box width value.
    # Data prep
    variable_names = list(dictionary.keys())  # Extract variable names from the dictionary
    data_for_plot = {variable: dictionary[variable] for variable in variable_names}

    # Plotting Customizations
    colors = [c_str]
    line_width = 1.5  # Uniform line width
    outlier_marker = {"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 3}

    # Make figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Make box plots
    for idx, (variable, values) in enumerate(data_for_plot.items()):
        ax.boxplot(values, positions=[idx], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor='none', color=colors[idx % len(colors)], linewidth=line_width),
                   whiskerprops=dict(color=colors[idx % len(colors)], linewidth=line_width),
                   capprops=dict(color=colors[idx % len(colors)], linewidth=line_width),
                   medianprops=dict(color=colors[idx % len(colors)], linewidth=line_width),
                   flierprops=outlier_marker)

    # Aesthetics
    ax.set_xticks(range(len(variable_names)))
    ax.set_xticklabels(variable_names, fontsize=22)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel(y_label, fontsize=22)
    plt.tight_layout()
    plt.savefig(y_label + ' Single Instability Measures Box Plots.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


def compare_boxplot(dictionary_list, y_label):

    # Data prep
    variable_names = list(dictionary_list[0].keys()) # Extract variable names from any of the dictionaries
    data_for_plot = {variable: [d[variable] for d in dictionary_list] for variable in variable_names}

    # Plotting Customizations
    colors = ['red', 'blue', 'green'] # Colors for data dictionaries
    line_width = 1.5  # Uniform line width
    gap = 1.0  # Gap between groups of variables, 1.5
    outlier_marker = {"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 3}

    ## Make figure
    fig, ax = plt.subplots(figsize=(15, 6))  # Adjusted figure size (15,6 )for 5variables for 4 variables ()

    n_dicts = len(dictionary_list) # Grab positions
    n_variables = len(variable_names)
    positions = []

    for i in range(n_variables):
        start = i * (n_dicts + gap)
        positions.extend([start + j for j in range(n_dicts)])

    # Make box plots
    for idx, (variable, values) in enumerate(data_for_plot.items()):
        for i, value in enumerate(values):
            ax.boxplot(value, positions=[positions[idx * n_dicts + i]], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='none', color=colors[i % len(colors)], linewidth=line_width),
                       whiskerprops=dict(color=colors[i % len(colors)], linewidth=line_width),
                       capprops=dict(color=colors[i % len(colors)], linewidth=line_width),
                       medianprops=dict(color=colors[i % len(colors)], linewidth=line_width),
                       flierprops=outlier_marker
                       )

    # Aesthetics
    ax.set_xticks([(n_dicts - 1) / 2 + i * (n_dicts + gap) for i in range(n_variables)])
    ax.set_xticklabels(variable_names, fontsize=22)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel(y_label, fontsize=22)
    plt.tight_layout()
    plt.savefig('Comparative Instability Measures Box Plots.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


def compare_kde(dictionary, x_label):

    # Make fig
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['red', 'blue', 'green']
    for idx, (label, values) in enumerate(dictionary.items()):
        sns.kdeplot(values, ax=ax, label=label, color=colors[idx % len(colors)], lw=1.0)

    # Aesthetics
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    # ax.legend(loc='upper center',  bbox_to_anchor=(0.5, -0.23), ncol=len(data_kde), fontsize=9) # fancybox=True, shadow=True,
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('Comparative KDE Plots ' + x_label + '.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return