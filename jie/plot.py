#!/usr/bin/env python

import numpy as np
import pandas as pd
import copy

from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import nan_euclidean_distances

def calc_med_dist_mat(pts_list, n_hyb):
    '''
    Input:
        pts_list : [list]
            list of DataFrames, each a table of spatial coordinates
        n_hyb : int
            number of genomic loci imaged for given chromosome
    Output:
        med_dist_mat : [ndarray]
            (n_hyb x n_hyb) median distance matrix of all chromatin fibers passed as input
    '''  
    # empty array to store all chromatin fiber distance matrices
    all_chr_dist = np.zeros((len(pts_list), n_hyb, n_hyb))
    
    # for every chromatin fiber
    for i, pts in enumerate(pts_list):
        
        # select columns
        chr_pts = copy.deepcopy(pts[['z_hat', 'y_hat', 'x_hat', 'hyb']])

        # create empty coordinate table
        chr_full = np.empty((n_hyb, 3))
        chr_full[:] = np.nan
        
        # append observed coordinates
        for elem in chr_pts.values:
            idx = int(elem[-1])
            chr_full[idx, :] = elem[0:3]

        # calculate distance matrix of given fiber
        chr_dist = nan_euclidean_distances(chr_full)
        
        # save distance matrix of given fiber
        all_chr_dist[i] = chr_dist
    
    # calculate median distance matrix among all fibers
    med_dist_mat = np.nanmedian(all_chr_dist, axis = 0)
    
    return med_dist_mat


def calc_gen_dist_mat(df_refgen, chosen_chrom):
    '''
    Input: 
        df_refgen : [DataFrame]
            reference table of genomic distances
        chosen_chrom : str
            chosen chromosome for analysis (format: chr{int})
    Output: 
        gen_mat : [ndarray]
            (n x n) matrix of pairwise genomic distances separating loci (where n = number of loci)
    '''
    # grab starting genomic coordinates
    start_coords = np.expand_dims(df_refgen[df_refgen['Chrom'] == chosen_chrom]['Start'].to_numpy(), 1)
    
    # find pairwise genomic distances
    gen_mat = squareform(pdist(start_coords))
    
    return gen_mat


def proximity_cluster(pts_list, gene_dist):
    '''
    Input:
        pts_list : [list]
            list of DataFrames, each a table of spatial coordinates
        gene_dist : [list]
            reference genomic distances between locis imaged on given chr
    Output:
        pts_clustered : [list]
            list of DataFrames, each a table of spatially-proximal coordinates
    '''
    pts_clustered = []
    
    for pts in pts_list:

        _df = copy.deepcopy(pts)

        # cluster
        print("eps", 13)
        dbsc = DBSCAN(eps = 13, min_samples = 10).fit(_df.loc[:, ['x_hat', 'y_hat', 'z_hat']].values)
        labels = np.abs(dbsc.labels_)
        _df['cluster'] = labels
        
        # for every spatial cluster
        for _, group in _df.groupby('cluster'):

            _df_grp = copy.deepcopy(group)
            
            # save
            pts_clustered.append(_df_grp)
            
    return pts_clustered


def power_func(x, a, b):
    '''   
    Input:
        x : float
        a : float
        b : float
    Output:
        a*x^b : float
    '''
    return a*np.power(x, b)


def fit_power(gen_mat, spa_mat):
    
    popt, pcov = curve_fit(power_func, 
                           gen_mat, 
                           spa_mat,
                           p0=[0,0],
                           bounds=(-np.inf, np.inf),
                           absolute_sigma = True)

    # calculate standard deviation of parameters
    stdevs = np.sqrt(np.diag(pcov))
    
    return popt, pcov, stdevs

