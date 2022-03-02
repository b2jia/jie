#!/usr/bin/env python

import numpy as np
import pandas as pd
import copy
import random

def cartesian_esqsum(left, right):
    '''
    Input:
        left : [DataFrame]
            table of l (sz, sy, sx) errors of starting nodes
        right : [DataFrame]
            table of r (sz, sy, sx) errors of ending nodes
    Output:
        a : [ndarray]
            (l+r) x 3 array of element-wise-squared cartesian sum
            eg. (sz_r**2 + sz_l**2, sy_r**2 + sy_l**2, sx_r**2 + sx_l**2)
    ''' 
    
    if not isinstance(left, pd.core.frame.DataFrame):
        raise TypeError('The first input must be a pd.DataFrame.')
    if not isinstance(right, pd.core.frame.DataFrame):
        raise TypeError('The second input must be a pd.DataFrame.')
    if left.shape[1] != 3 or right.shape[1] != 3:
        raise Exception('The input dimension is not 3-D (n x 3).')
        
    la, lb = len(left), len(right)        
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
    a = np.column_stack([left.values[ia2.ravel()] ** 2, right.values[ib2.ravel()] ** 2])
    a = a[:, :left.shape[1]] + a[:, right.shape[1]:]
    
    return a

def cartesian_sqdiff(left, right):
    '''
    Input:
        left : [DataFrame]
            table of l (z, y, x) coordinates of starting nodes
        right : [DataFrame]
            table of r (z, y, x) coordinates of ending nodes
    Output:
        a : [ndarray]
            (l+r) x 3 array of squared cartesian difference
            eg. ((z_r - z_l)**2, (y_r - y_l)**2, (x_r - x_l)**2)
    '''
    
    if not isinstance(left, pd.core.frame.DataFrame):
        raise TypeError('The first input must be a pd.DataFrame.')
    if not isinstance(right, pd.core.frame.DataFrame):
        raise TypeError('The second input must be a pd.DataFrame.')
    if left.shape[1] != 3 or right.shape[1] != 3:
        raise Exception('The input dimension is not 3-D (n x 3).')
        
    la, lb = len(left), len(right)
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
    a = np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]])
    a = a[:, :left.shape[1]] - a[:, right.shape[1]:]
    return a**2

def cartesian_diff(left, right):
    '''
    Input:
        left : [DataFrame]
            series of genomic order (t) for (L) starting nodes
        right : [DataFrame]
            series of genomic order (t) for (R) ending nodes
    Output:
        a : [ndarray]
            (L+R) x 1 array of cartesian difference btwn genomic order
            eg. [(t_r1 - t_l1), (t_r2 - t_l1), ..., (t_rR - t_lL)]
    '''
    
    if not isinstance(left, pd.core.series.Series):
        raise TypeError('The first input must be a pd.Series.')
    if not isinstance(right, pd.core.series.Series):
        raise TypeError('The second input must be a pd.Series.')
        
    la, lb = len(left), len(right)
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
    a = np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]])
    a = a[:, 1] - a[:, 0]
    return a

def find_loci_dist(gene_dist, nm_per_bp, pixel_dist):
    '''
    Input:
        gene_dist : [list]
            reference genomic distances between locis imaged on given chr
            eg. [0, 5kb, 10kb, 15kb ...]
        nm_per_bp : float
            length scale of chromosome
        pixel_dist : float
            nm of one pixel
    Output:
        loci_dist : [ndarray]
            n_loci x n_loci array of pairwise expected spatial distance given genomic distance
            eg. [[0, 5kb*nm_per_bp/pixel_dist, 10kb*nm_per_bp/pixel_dist,  ...]
                 [5kb*nm_per_bp/pixel_dist,  0, 5kb*nm_per_bp/pixel_dist,  ...]
                 [10kb*nm_per_bp/pixel_dist, 5kb*nm_per_bp/pixel_dist, 0, ...]]
    '''    
    if not isinstance(gene_dist[0], (int, float, np.int64, np.float64)):
        raise TypeError('gene_dist must be an iterable of numerical values.')
    if not isinstance(nm_per_bp, (int, float, np.int64, np.float64)):
        raise TypeError('nm_per_bp must be a numerical value.')
    if nm_per_bp <= 0:
        raise ValueError('nm_per_bp must be non-negative.')
    if not isinstance(pixel_dist, (int, float, np.int64, np.float64)):
        raise TypeError('pixel_dist must be a numerical value.')
    if pixel_dist <= 0:
        raise ValueError('pixel_dist must be non-negative.')
  
     # create distance vector
    L_vector = np.array([(elem * nm_per_bp) / pixel_dist for elem in gene_dist])

    # generate dist mtx
    loci_dist = np.abs( np.array([L_vector]* len(L_vector)) - np.array([L_vector]*len(L_vector)).transpose() )
    
    return loci_dist

def check_lp_wgaps(left, right, l_p, loci_dist):
    '''
    Input:
        left : [DataFrame]
            series of genomic order (t) for (L) starting nodes
        right : [DataFrame]
            series of genomic order (t) for (R) ending nodes
        l_p : float
            l_p_bp * nm_per_bp / pixel_dist (pixels)
        loci_dist: [ndarray]
            n_loci x n_loci array of pairwise expected spatial distance given genomic distance
            eg. [[0, 5kb*nm_per_bp/pixel_dist, 10kb*nm_per_bp/pixel_dist,  ...]
                 [5kb*nm_per_bp/pixel_dist,  0, 5kb*nm_per_bp/pixel_dist,  ...]
                 [10kb*nm_per_bp/pixel_dist, 5kb*nm_per_bp/pixel_dist, 0, ...]]
    Output:
        a : [ndarray]
            (l+r) x 3 array of element-wise-squared cartesian sum
            eg. (sz_r**2 + sz_l**2, sy_r**2 + sy_l**2, sx_r**2 + sx_l**2)
    '''
    
    if not isinstance(left, pd.core.series.Series):
        raise TypeError('The first input must be a pd.Series.')
    if not isinstance(right, pd.core.series.Series):
        raise TypeError('The second input must be a pd.Series.')
    if not isinstance(l_p, (int, float, np.int64, np.float64)):
        raise TypeError('l_p must be a numerical value (eg. l_p_bp * nm_per_bp / pixel_dist).')
    if l_p > 1:
        raise ValueError('l_p must be in pixel distance (eg. l_p_bp * nm_per_bp / pixel_dist). Double check l_p_bp, nm_per_bp, and pixel_dist.')
    if np.max(left) > loci_dist.shape[0] or np.max(right) > loci_dist.shape[0]:
        raise IndexError('The locus index is out of bounds w.r.t. the loci_dist.')
        
    la, lb = len(left), len(right)
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
    a = np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]])
    locus_dist = np.array([loci_dist[int(i)][int(j)] for (i, j) in zip(a[:, 0], a[:, 1])])
    s_sq1 = np.array([2*l_p*elem/3 for elem in locus_dist])
    s_sq1[np.where(locus_dist < 2*l_p)] = (locus_dist[np.where(locus_dist < 2*l_p)]**2)/3
    
    return s_sq1

