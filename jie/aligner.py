#!/usr/bin/env python

import numpy as np
import pandas as pd
import igraph
import copy
import warnings
import random

from collections.abc import Iterable

from .utilities import (cartesian_esqsum, 
                        cartesian_sqdiff, 
                        cartesian_diff, 
                        check_lp_wgaps,
                        find_loci_dist)

def log_bond(l_p, ideal_l, r):
    '''
    Input:
        l_p: float
            persistence length
        ideal_l: float 
           distance of one bond length given bin_size
        r: float
            distance of observed bond
    Output:
        prob: float
            log probability of polymer segment
    '''
    if not isinstance(l_p, (int, float, np.int64, np.float64)):
        raise TypeError('l_p needs to be a numerical value.')
    if l_p <= 0:
        raise ValueError('l_p must be a positive value.')
    if not isinstance(ideal_l, (int, float, np.int64, np.float64)):
        raise TypeError('ideal_l needs to be a numerical value.')
    if ideal_l <= 0:
        raise ValueError('ideal_l must be a positive value.')
        
    # calculate uncertainty
    ## NB: l_p usually in pixels, ideal_l can be bp (unless corr_fac**2 --> penalty)
    s_sq = 2*l_p*ideal_l/3 
    
    # calculate bond probability
    prob = -np.log(np.power((2*np.pi*s_sq), -3/2)) + ((r**2)/(2*s_sq))
    
    return prob
    
    
def log_bond_vect(l_p_arr, ideal_l_arr, r_arr):
    '''
    Input:
        l_p_arr: [curr_hyb x next_hyb]
            2D array of persistence length
        ideal_l_arr: [curr_hyb x next_hyb] 
            2D array of ideal bond distances
        r_arr: [curr_hyb x next_hyb]
            2D array of observed bond distances
    Output:
        prob_arr: [curr_hyb x next_hyb] 
            2D array of log probability of polymer segments        
    '''
    if not isinstance(l_p_arr, np.ndarray):
        raise TypeError('l_p_arr needs to be an array.')
    if l_p_arr.dtype != float and l_p_arr.dtype != int:
        raise TypeError('l_p_arr needs to be an array of numerical values')
    if not np.all(l_p_arr > 0):
        raise ValueError('l_p_arr needs to be an array of positive numerical values.')
    if not isinstance(ideal_l_arr, np.ndarray):
        raise TypeError('ideal_l needs to be an array.')
    if ideal_l_arr.dtype != float and ideal_l_arr.dtype != int:
        raise TypeError('ideal_l_arr needs to be an array of numerical values')
    if not np.all(ideal_l_arr > 0):
        raise ValueError('ideal_l_arr needs to be an array of positive numerical values.') 
    if not isinstance(r_arr, np.ndarray):
        raise TypeError('r_arr needs to be an array.')
    if r_arr.dtype != float and r_arr.dtype != int:
        raise TypeError('r_arr needs to be an array of numerical values')  
    
    # calculate uncertainty
    ## NB: l_p_arr usually in pixels, ideal_l can be bp (unless corr_fac ** 2 --> penalty)
    s_sq_arr = (2/3)*np.multiply(l_p_arr, ideal_l_arr)
    
    # calculate bond probability
    prob_arr = -np.log(np.power((2*np.pi*s_sq_arr), -3/2)) + ((r_arr**2)/(2*s_sq_arr))
    
    return prob_arr


def cdf_thresh(gene_dist, l_p_bp):
    '''
    Input:
        gene_dist: [list]
            list of genomic distances relative to starting loci
            eg. [0, 5kb, 10kb, 15kb, ...]
        l_p_bp: float
            persistence length (bp)
    Output:
        total_score: float
            conformational distribution function (CDF) threshold score for for calling chr
    '''
    if not isinstance(gene_dist, Iterable):
        raise TypeError('gene_dist needs to be an iterable list of numerical values.')
        
    if not isinstance(gene_dist[0], (int, float, np.int64, np.float64)):
        raise TypeError('gene_dist needs to be an iterable list of numerical values.')

    if not np.all(np.diff(gene_dist) > 0):
        raise ValueError('gene_dist must be an ascending sorted list of numerical values.')

    if not np.all(np.array(gene_dist) >= 0):
        raise ValueError('gene_dist must be positive numerical values.')
        
    if not isinstance(l_p_bp, (int, float, np.int64, np.float64)):
        raise TypeError('l_p_bp needs to be a numerical value')
        
    if l_p_bp < 0:
        raise ValueError('l_p_bp needs to be greater than 0.')
    
    # scramble genomic positions
    gene_dist_scrambled = copy.deepcopy(gene_dist)
    random.shuffle(gene_dist_scrambled) # in place
    
    if np.all(gene_dist_scrambled == gene_dist):
        warnings.warn('gene_dist not scrambled during CDF threshold estimation.')
        
    # calculate new genomic distances
    gene_delta = [y - x for x,y in zip(gene_dist,gene_dist[1:])]
    gene_delta_scrambled = [ np.abs(y - x) for x,y in zip(gene_dist_scrambled,gene_dist_scrambled[1:])]
    
    # calculate path
    total_score = 0
    for ideal_l, r in zip(gene_delta, gene_delta_scrambled):
        total_score += log_bond(l_p_bp, ideal_l, r)
    
    return total_score

    
def edge_penalty(skips, l_p_bp, corr_fac, bin_size):
    '''
    Input:
        skips: [curr_hyb x next_hyb]
            2D array of degrees-of-separation between nodes
        l_p_bp: float 
            persistence length (bp)
        corr_fac: float 
            scale genomic dist (bp) into pixels (e.g nm_per_bp / pixel_dist)
        bin_size: float 
            median genomic distance interval (bp)
    Output:
        penalty: [curr_hyb x next_hyb] 
            2D array of penalty per transition edge weight
    '''
    if not isinstance(skips, np.ndarray) or skips.dtype != int:
        raise TypeError('skips needs to be a 2D numpy array of integers.')       
    
    if not isinstance(l_p_bp, (int, float, np.int64, np.float64)):
        raise TypeError('l_p_bp needs to be a numerical value.')
    
    if l_p_bp < 0:
        raise ValueError('l_p_bp needs to be a positive numerical value.')
        
    if not isinstance(corr_fac, (int, float, np.int64, np.float64)):
        raise TypeError('corr_fac needs to be a numerical value.')
    
    if corr_fac < 0:
        raise ValueError('corr_fac needs to be a positive numerical value.')
        
    if not isinstance(bin_size, (int, float, np.int64, np.float64)):
        raise TypeError('bin_size needs to be a numerical value.')
    
    if bin_size < 0:
        raise ValueError('bin_size needs to be a positive numerical value.')
    
    # broadcast persistence length
    l_p_arr = np.full(skips.shape, l_p_bp*corr_fac)
    
    # calculate estimated bond lengths
    ideal_l_mult_arr = (skips + 1)*bin_size*corr_fac
    
    # calculate single bond length
    ideal_l_sing_arr = np.full(skips.shape, bin_size*corr_fac)
    
    # evaluate relative length w.r.t. single bond length
    # NB: (additional corr_fac --> better behaviour)
    ratio = (log_bond_vect(l_p_arr, ideal_l_mult_arr, ideal_l_mult_arr) / 
             log_bond_vect(l_p_arr, ideal_l_sing_arr, ideal_l_sing_arr) )
    
    # calculate relative penalty
    penalty = np.divide(1.01*(skips+1), ratio)
    
    # adjacent bonds not penalized
    penalty[skips == 0] = 1
    
    return penalty


def edge_weights(pts_clr_curr, 
                 pts_clr_next, 
                 bin_size, 
                 l_p_bp,
                 nm_per_bp, 
                 pixel_dist, 
                 theta, 
                 loci_dist,
                 lim_min_dist = True):
    '''
    Input:
        pts_clr_curr: [DataFrame]
            table of spatial coordinates + metadata of current nodes
        pts_clr_next: [DataFrame]
            table of spatial coordinates + metadata of reachable nodes
        bin_size: float
            median genomic distance interval (bp)
        l_p_bp: float
            persistence length (bp)
        nm_per_bp: float
            scaling factor converting genomic distance (bp) -> spatial distance (nm)
        pixel_dist: float
            pixel size (nm)
        theta: float
            bond angle
        loci_dist: [num_hyb x num_hyb] 
            2D array of pairwise expected spatial distance given genomic distance
        lim_min_dist : boolean
            penalize successively choosing most proximal spot 
    Output:
        trans_prob: [curr_hyb x next_hyb] 
            2D array of transition edge weights
    '''
    if not isinstance(pts_clr_curr, pd.core.frame.DataFrame) or not isinstance(pts_clr_next, pd.core.frame.DataFrame):
        raise TypeError('Both pts_clr_curr and pts_clr_next must be pandas DataFrames')
        
    if not set(['x_hat', 'y_hat', 'z_hat', 'hyb', 'sig_x', 'sig_y', 'sig_z']).issubset(pts_clr_curr.columns) or \
       not set(['x_hat', 'y_hat', 'z_hat', 'hyb', 'sig_x', 'sig_y', 'sig_z']).issubset(pts_clr_next.columns):
        raise KeyError('pts_clr_curr and pts_clr_next must have the following columns: [x_hat, y_hat, z_hat, hyb, sig_x, sig_y, sig_z]')
                       
    if pts_clr_next['hyb'].size > 0:
        if max(pts_clr_next['hyb']) > loci_dist.shape[0]:
            raise IndexError('hyb index out of bounds with respect to loci_dist. Check if the correct reference genome is being used.')

    if not pts_clr_curr.hyb.is_monotonic_increasing or not pts_clr_next.hyb.is_monotonic_increasing:
        raise ValueError('hyb in both pts_clr_curr and pts_clr_next must be sorted in ascending order.')
                
    if not all(isinstance(x, (int, float, np.int64, np.float64)) for x in [bin_size, l_p_bp, nm_per_bp, pixel_dist, theta]) or \
       not all(x >= 0 for x in [bin_size, l_p_bp, nm_per_bp, pixel_dist, theta]):
        raise ValueError('bin_size, l_p_bp, nm_per_bp, pixel_dist, theta must be all positive numerical values.')
        
    # grab output shape
    shape = (pts_clr_curr.shape[0], pts_clr_next.shape[0])
    
    ##### OBSERVED ######
    # Calculate observed sq distance
    sq_diff = cartesian_sqdiff(pts_clr_curr[['z_hat', 'y_hat', 'x_hat']], 
                               pts_clr_next[['z_hat', 'y_hat', 'x_hat']] )
    r_sq = np.sum(sq_diff, axis = 1)
    #####################
    
    ##### EXPECTED ######
    # Determine distances closer than expected
    separation = cartesian_diff(pts_clr_curr['hyb'], pts_clr_next['hyb'])
    # Disregard immediate linkage between adjacent monomers
    separation[separation <= 1] = 0

    # Calculate expected pixel distance sq
    exp_len_sq = (separation * bin_size * nm_per_bp * (np.cos(theta)**separation ) / pixel_dist) **2
    # Split expected sq distance along each axis
    exp_len_sq_singaxis = np.array([(exp_len_sq)/3, (exp_len_sq)/3, (exp_len_sq)/3]).T
    #####################
    
    ### OBS -> X, Y, Z ###
    # Replace any observed distance that are too close (may not be necessary)
    if lim_min_dist:
        sq_diff[np.where(r_sq < exp_len_sq)] = exp_len_sq_singaxis[np.where(r_sq < exp_len_sq)]
    
    # Split observed sq distance into indiv axis
    z_sq = np.reshape(sq_diff[:, 0], shape)
    y_sq = np.reshape(sq_diff[:, 1], shape)
    x_sq = np.reshape(sq_diff[:, 2], shape)
    ######################
    
    #### UNCERTAINTY #####
    # Calculate uncertainty d.t. contour
    s_sq_1 = check_lp_wgaps(pts_clr_curr['hyb'], 
                            pts_clr_next['hyb'], 
                            l_p_bp * nm_per_bp / pixel_dist, 
                            loci_dist)
    s_sq_1 = np.reshape(s_sq_1, shape)     

    # Calculate uncertainty from Gaussian fitting of each point
    s_sq_2 = cartesian_esqsum(pts_clr_curr[['sig_z', 'sig_y', 'sig_x']], 
                              pts_clr_next[['sig_z', 'sig_y', 'sig_x']])
    # Split into each axis
    s_sq_2z = np.reshape(s_sq_2[:, 0], shape)
    s_sq_2y = np.reshape(s_sq_2[:, 1], shape)
    s_sq_2x = np.reshape(s_sq_2[:, 2], shape)

    # Calculate total uncertainty
    s_sq_z_tot = s_sq_1 + s_sq_2z
    s_sq_y_tot = s_sq_1 + s_sq_2y
    s_sq_x_tot = s_sq_1 + s_sq_2x
    ######################
    
    ### TRANSITION EDGE WEIGHTS ###
    # Calculate constant term
    const = np.multiply( (1/ np.sqrt(2*np.pi*s_sq_z_tot)), (1/ np.sqrt(2*np.pi*s_sq_y_tot)))
    const = np.multiply( const,                            (1/ np.sqrt(2*np.pi*s_sq_x_tot)))
    
    # Calculate exp term
    exp = np.add( z_sq/(2*s_sq_z_tot) , y_sq/(2*s_sq_y_tot))
    exp = np.add( exp,                  x_sq/(2*s_sq_x_tot))
    
    # Calculate negative log prob of Gaussian chain link
    trans_prob = np.add(-np.log(const), exp)
    ################################
    
    assert trans_prob.shape == shape 
    
    return trans_prob


def boundary_init(trans_mat, 
                  loci_dist, 
                  l_p_bp,
                  corr_fac,
                  n_colours, 
                  cell_pts, 
                  exp_stretch, 
                  stretch_factor, 
                  lim_init_skip, 
                  init_skip,
                  end_skip):
    '''
    Input:
        trans_mat: [ndarray]
            2D adjacency matrix
        loci_dist: [num_hyb x num_hyb ndarray] 
            2D array of pairwise expected spatial distance given genomic distance
        l_p_bp: float
            persistence length (bp)
        corr_fac: float
            scale genomic dist (bp) into pixels (e.g nm_per_bp / pixel_dist)
        n_colours: float
            number of loci imaged for given chr
        cell_pts: [DataFrame]
            table of spatial coordinates of all nodes + metadata in cell
        exp_stretch: float
            expected bond extension
        stretch_factor: float
            allowable bond extension
        lim_init_skip: boolean
            limit number of skips at graph source and sink
        init_skip: int
            index of hyb to be skipped to from graph source
        end_skip: int
            index of hyb to allow skips to graph sink
    Output:
        trans_mat_pad: [(num_hyb + 2) x (num_hyb + 2)] 
            2D array of transition edge weights, padded with initial and terminal gap penalties
    '''
    if not isinstance(cell_pts, pd.core.frame.DataFrame):
        raise TypeError('cell_pts needs to be a pandas DataFrame.')
        
    if not set(['hyb', 'CurrIndex']).issubset(cell_pts.columns):
        raise KeyError('cell_pts must have the following columns: [hyb, CurrIndex]')
                       
    if not cell_pts['hyb'].is_monotonic_increasing:
        raise IndexError('cell_pts[hyb] not sorted.')
        
    if not isinstance(trans_mat, np.ndarray) or not isinstance(loci_dist, np.ndarray):
        raise TypeError('Adjacency matrix (trans_mat) and genomic distance (loci_dist) must both be numpy arrays.')
                
    if not loci_dist.shape[0] == loci_dist.shape[1] == n_colours:
        raise ValueError('Dimension mismatch: loci_dist, trans_mat must both be n_colours x n_colours arrays.')
        
    if not all(isinstance(x, (int, float, np.int64, np.float64)) for x in [l_p_bp, corr_fac, exp_stretch, stretch_factor]):
        raise TypeError('l_p_bp, corr_fac, exp_stretch, stretch_factor must be all positive numerical values.')
        
    if not all(x >= 0 for x in [l_p_bp, corr_fac, exp_stretch, stretch_factor]):
        raise ValueError('l_p_bp, corr_fac, exp_stretch, stretch_factor must be all positive numerical values.')
        
    if not isinstance(init_skip, (int, np.int64)) or not isinstance(end_skip, (int, np.int64)):
        raise TypeError('init_skip, end_skip must be integer indeces.')
        
    if not 0 <= init_skip < end_skip:
        raise ValueError('The following must be satisfied: 0 <= init_skip < end_skip .')      
        
    # Pad transition matrix with source and sink
    trans_mat_pad = np.zeros((trans_mat.shape[0]+2, trans_mat.shape[1]+2))

    # dummy probability
    small_num = 1e-26

    # get ideal genomic distance intervals
    bp_intervals = [loci_dist[0][i] - loci_dist[0][i-1] for i in range(1, len(loci_dist[0]))]

    # add "border" transition probabilities
    for h in range(n_colours):

        # use subset index to subset dataframe
        pts_clr_curr = cell_pts.loc[cell_pts['hyb']==h]

        # parse row, col indeces 
        row_idx = pts_clr_curr['CurrIndex'].values.astype(int) + 1 # +1 to shift over from source
        col_idx = [(elem+1)*trans_mat_pad.shape[0]-1 for elem in row_idx] #+1 to shift over from sink

        # calculate ideal bond prob
        # NB: imaginary linkages based on genomic dist (better behaviour)
        ideal_l_arr_row = [log_bond(l_p_bp*corr_fac, exp_stretch*interval, 
                                    stretch_factor*interval) for interval in bp_intervals[:h]]
        stretch_bond_row = np.sum(ideal_l_arr_row)        
        ideal_l_arr_col = [log_bond(l_p_bp*corr_fac, exp_stretch*interval, 
                                    stretch_factor*interval) for interval in bp_intervals[h:]]
        stretch_bond_col = np.sum(ideal_l_arr_col)
        
        # calculate values of transition prob
        if h == 0:
            prob_row = small_num
        elif h != 0:
            prob_row = stretch_bond_row
        else:
            prob_row = None

        if h == n_colours - 1:
            prob_col = small_num
        elif h != n_colours -1:
            prob_col = stretch_bond_col
        else:
            prob_col = None

        # update transition matrix
        if prob_row:
            if lim_init_skip == True:
                if h <= init_skip:
                    np.put(trans_mat_pad, row_idx, prob_row)
                else:
                    np.put(trans_mat_pad, row_idx, [0, ] * len(row_idx))
            else:
                np.put(trans_mat_pad, row_idx, prob_row)

        if prob_col:
            if lim_init_skip == True:
                if h >= end_skip:
                    np.put(trans_mat_pad, col_idx, prob_col)
                else:
                    np.put(trans_mat_pad, col_idx, [0, ] * len(col_idx))
            else:
                np.put(trans_mat_pad, col_idx, prob_col)

    # fill in transition matrix "center"
    trans_mat_pad[1:-1, 1:-1] = trans_mat

    # replace nan's as inaccessible edges
    trans_mat_pad[np.isnan(trans_mat_pad)] = 0
    
    return trans_mat_pad


def find_chr(cell_pts_input,
             gene_dist,
             bin_size,
             nm_per_bp = .0004,
             pixel_dist = 100.,
             l_p_bp = 150., 
             stretch_factor = 1.2, 
             exp_stretch = 1., 
             num_skip = 7, 
             total_num_skip_frac = 0.7,
             init_skip_frac = 0.15,
             theta = np.pi/20, 
             norm_skip_penalty = True,
             lim_init_skip = True,
             lim_min_dist = True,
            ):
    
    '''
    From a DataFrame cell_pts_input, builds a graph where transition probs based on freely jointed chain model
    of DNA and returns the most likely polymer path.

    NB: prob of emission over one path P(X,Π) = conformational distribution function (product of N bonds)
    NB2: prob of emission over all paths P(X) = partition function

    Input:
        cell_pts_input: [DataFrame] 
            spatial coordinates + metadata of locis detected in one nucleus
        gene_dist : [list]
            reference genomic distances between locis imaged on given chr
        bin_size : float
            median base pair interval between genomic loci
        nm_per_bp : float
            length scale of chromosome
        pixel_dist : float
            nm of one pixel
        l_p_bp: float
            persistence length of DNA (bp)
        stretch_factor: float
            fraction of max allowable ideal bond length to determine skip
        exp_stretch: float
            fraction of ideal bond length expected to span two loci
        num_skip: int
            number of locis allowed to skip for one step
        total_num_skip_frac: float
            fraction of total locis allowed to skip for entire path
        init_skip_frac: float
            fraction of total locis allowed to skip at graph source and sink
        theta: float
            bond angle
        lim_init_skip: boolean
            limit number of skips at graph source and sink
        lim_min_dist : boolean
            penalize successively skipping
    Output:
        trans_mat_pad: [ndarray] 
            adjacency matrix of polymer model
        shortest_path: [list] 
            shortest path (most likely chromosome)
        shortest_path_length: float 
            conformational distribution function (CDF) of most likely polymer
    '''
    
    if not stretch_factor > exp_stretch >= 1:
        raise ValueError('stretch_factor must be greater than exp_stretch, which must be greater or equal to 1.')
        
    if not bin_size/l_p_bp >= 10:
        raise ValueError('bin_size (countour length) must be >> l_p_bp. Double check the input persistence length and bin size.')
        
    ## Define constants ##
    n_colours = len(gene_dist)
    total_num_skip = int(total_num_skip_frac*len(gene_dist)) 
    corr_fac = nm_per_bp / pixel_dist
    l_p = l_p_bp * nm_per_bp / pixel_dist # persistence length in pixels
    init_skip = int(init_skip_frac*n_colours)
    end_skip = int((1-init_skip_frac)*n_colours)
    loci_dist = find_loci_dist(gene_dist = gene_dist,
                               nm_per_bp = nm_per_bp,
                               pixel_dist = pixel_dist)
    gdintervals = [loci_dist[0][i] - loci_dist[0][i-1] for i in range(1, len(loci_dist[0]))]
    
    if not np.all([elem/l_p >= 10 for elem in gdintervals]):
        raise ValueError('Spatial distance estimated from genomic intervals separating loci (contour length) must be >> l_p (pixel dist). Double check the input persistence length (l_p_bp) and reference genome intervals (gene_dist).')

    # make copy of input dataframe
    cell_pts = copy.deepcopy(cell_pts_input)
    
    # check if sorted
    try:
        assert cell_pts['hyb'].is_monotonic_increasing   
    except AssertionError:
        cell_pts = cell_pts.sort_values(by='hyb')  

    # add current index
    cell_pts.reset_index(inplace=True, drop = True)
    cell_pts['CurrIndex'] = cell_pts.index

    # create transition matrix
    n_states =  cell_pts.shape[0]  
    trans_mat = np.zeros((n_states, n_states))

    for i in set(cell_pts['hyb']):

        # grab nodes of curr hyb and reachable hyb
        pts_clr_curr = cell_pts.loc[cell_pts['hyb']==i]
        pts_clr_next = cell_pts.loc[cell_pts['hyb'].between(i, i+num_skip, inclusive = False)]

        # grab node indeces in adjacency matrix
        ## NB: rows --> starting vertices | cols --> ending vertices
        rows = pts_clr_curr['CurrIndex'].values
        cols = pts_clr_next['CurrIndex'].values

        # convert node indeces to position in transition matrix
        trans_idx = (np.array([n_states*rows, ] * len(cols)).T + np.array(cols)).flatten().astype(int)

        # find degrees of separation between nodes
        next_hyb = i+1
        skips = np.array([pts_clr_next['hyb'].values,] * pts_clr_curr.shape[0] ) - np.min([next_hyb, n_colours-1])

        # calculate edge penalties
        penalty = edge_penalty(skips, l_p_bp, corr_fac, bin_size)

        # calculate edge weights
        trans_prob = edge_weights(pts_clr_curr, pts_clr_next, 
                                  bin_size, l_p_bp, 
                                  nm_per_bp, pixel_dist, 
                                  theta, loci_dist, lim_min_dist)

        # check if penalty to be applied
        if norm_skip_penalty == True:
            # apply skipping penalty
            if len(trans_prob) > 0:
                trans_prob = np.multiply(trans_prob, penalty)

        # update transition matrix
        np.put(trans_mat, trans_idx, trans_prob)

    # calculate initial and terminal gap penalties
    trans_mat_pad = boundary_init(trans_mat=trans_mat,
                                  loci_dist=loci_dist, 
                                  l_p_bp=l_p_bp,
                                  corr_fac=corr_fac,
                                  n_colours=n_colours,
                                  cell_pts=cell_pts,
                                  exp_stretch=exp_stretch,
                                  stretch_factor=stretch_factor,
                                  lim_init_skip=lim_init_skip,
                                  init_skip=init_skip,
                                  end_skip=end_skip)
    
    if not np.all(trans_mat_pad >= 0):
        raise ValueError('Edge weights cannot be negative. Double check persistence length (l_p_bp), bin size (bin_size), distance parameter (nm_per_bp) and pixel distance (pixel_dist).')
    
    # create discrete state space model
    G = igraph.Graph.Adjacency((trans_mat_pad > 0).tolist())

    # add edge weights
    G.es['weight'] = trans_mat_pad[trans_mat_pad.nonzero()]

    # check boundary conditions
    if len(np.unique(cell_pts_input.hyb)) < len(gene_dist) - total_num_skip:
        return trans_mat_pad, [], -1
    else:
        try:
            # find shortest path (Dijkstra)
            shortest_path_length = G.shortest_paths(source = 0, 
                                                    target = trans_mat_pad.shape[0]-1, 
                                                    weights = 'weight')[0][0]
            ## NB: path is written in index of FUTURE cell_pts dataframe (subsetted after iterative subtraction)
            shortest_path = [elem-1 for elem in G.get_shortest_paths(0, to = trans_mat_pad.shape[0]-1, 
                                                                     weights = 'weight')[0][1:-1]]
            if len(shortest_path) < len(gene_dist) - total_num_skip:
                return trans_mat_pad, [], -1
        except:
            return trans_mat_pad, [], -1

    return trans_mat_pad, shortest_path, shortest_path_length


def find_all_chr(cell_pts_input,
                 gene_dist,
                 bin_size,
                 nm_per_bp = .0004,
                 num_skip = 7,
                 total_num_skip_frac = 0.7,
                 init_skip_frac = 0.15,
                 pixel_dist = 100.,
                 l_p_bp = 150., 
                 stretch_factor = 1.2, 
                 exp_stretch = 1., 
                 theta = np.pi/20, 
                 norm_skip_penalty = True,
                 lim_init_skip = True,
                 max_iter = 6):
    '''
    From a DataFrame cell_pts_input, iteratively builds graphs where transition probs based on 
    freely jointed chain model of DNA. Each iteration finds a most likely path, upon which
    nodes on the path are subtracted from the graph and a new graph is built for the next
    iteration.

    Input:
        cell_pts_input: [DataFrame] 
            spatial coordinates + metadata of locis detected in one nucleus
        gene_dist : [list]
            reference genomic distances between locis imaged on given chr
        bin_size : float
            median base pair interval between genomic loci
        nm_per_bp : float
            length scale of chromosome
        num_skip: int
            number of locis allowed to skip for one step
        total_num_skip_frac: float
            fraction of total locis allowed to skip for entire path
        init_skip_frac: float
            fraction of total locis allowed to skip at graph source and sink
        pixel_dist : float
            nm of one pixel
        l_p_bp: float
            persistence length of DNA (bp)
        stretch_factor: float
            fraction of max allowable ideal bond length to determine skip
        exp_stretch: float
            fraction of ideal bond length expected to span two loci
        theta: float
            bond angle
        lim_init_skip: boolean
            limit number of skips at graph source and sink
        max_iter : int
            max iteration for iterative path finding
    Output:
        all_put_chr: [list]
            list of DataFrames, each an orthogonal set of coords belonging to a chromatin fiber
    '''
    # define constants
    n_colours = len(gene_dist)
    cd_func_scrambled = cdf_thresh(gene_dist, l_p_bp) # pass only genomic dist, not pixel dist
    total_num_skip = int(total_num_skip_frac * len(gene_dist))
    
    # copy dataframe
    cell_pts = copy.deepcopy(cell_pts_input)
    
    # check hyb rnd is int
    try:
        assert cell_pts['hyb'].dtype == int

    except AssertionError:
        warnings.warn("Hybridization rounds should be integers.")
        cell_pts['hyb'] = cell_pts['hyb'].astype(int)
        
    # check if sorted
    try:
        assert cell_pts['hyb'].is_monotonic_increasing
    
    except AssertionError:
        warnings.warn("Input DataFrame needs to be sorted in order of imaged loci ('hyb').")
        cell_pts = cell_pts.sort_values(by='hyb')    

    # drop index
    cell_pts.reset_index(inplace=True, drop = True)
    
    all_put_chr = []   
    last_len = cell_pts.shape[0]
    last_cdf = 0
    last_path = None
    
    for iteration in range(max_iter):
        
        if iteration == 0:
            if last_len <= 0:
                break
        else:
            if (len(cell_pts) <= 0) or \
               (len(cell_pts) == last_len) or \
               (last_cdf >= cd_func_scrambled) or \
               (len(last_path) < n_colours - total_num_skip):
                break

        _, path, cd_func = find_chr(cell_pts, 
                                    gene_dist = gene_dist,
                                    bin_size = bin_size,
                                    nm_per_bp = nm_per_bp, 
                                    stretch_factor = stretch_factor, 
                                    num_skip = num_skip, 
                                    theta=theta,
                                    total_num_skip_frac = total_num_skip_frac,
                                    init_skip_frac = init_skip_frac,
                                    lim_init_skip = True)

        # grab fiber pts
        put_chr = cell_pts.iloc[[elem for elem in path]]
        
        # threshold for visitation length and physical likelihood
        if (len(path) >= n_colours - total_num_skip) and (cd_func < cd_func_scrambled):
            
            # save chr
            all_put_chr.append(put_chr)

        # measure length
        last_cdf = cd_func
        last_len = len(cell_pts)
        last_path = path
        
        # prune nodes
        cell_pts = cell_pts.iloc[[i for i in range(cell_pts.shape[0]) if i not in path]]

    return all_put_chr

