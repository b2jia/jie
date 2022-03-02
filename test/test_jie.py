import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import copy
import random

from jie.utilities import (cartesian_esqsum,
                           cartesian_sqdiff,
                           cartesian_diff,
                           check_lp_wgaps, 
                           find_loci_dist)

from jie.aligner import (log_bond,
                         log_bond_vect,
                         cdf_thresh,
                         edge_penalty,
                         edge_weights,
                         boundary_init,
                         find_chr,
                         find_all_chr)


class TestCartesianEsqSum(unittest.TestCase):
    
    def setUp(self):
        self.left = pd.DataFrame([[1, 1, 1], [3, 3, 3]])
        self.right = pd.DataFrame([[1, 1, 1], [4, 4, 4]])
        
    def test_cartesian_esqsum_output(self):
        self.assertEqual(cartesian_esqsum(self.left, self.right).tolist(), 
                         [[ 2,  2,  2], [17, 17, 17], [10, 10, 10], [25, 25, 25]])
    
    def test_wrong_first_input_type(self):
        left = np.array([[1, 1, 1], [3, 3, 3]])
        self.assertRaises(TypeError, cartesian_esqsum, *(left, self.right))
        
    def test_wrong_second_input_type(self):
        right = np.array([[1, 1, 1], [4, 4, 4]])
        self.assertRaises(TypeError, cartesian_esqsum, *(self.left, right))
    
    def test_wrong_first_input_dim(self):
        left = pd.DataFrame([[1, 3], [1, 3], [1, 3]])
        self.assertRaises(Exception, cartesian_esqsum, *(left, self.right))
    
    def test_wrong_second_input_dim(self):
        right = pd.DataFrame([[1, 3], [1, 3], [1, 3]])
        self.assertRaises(Exception, cartesian_esqsum, *(self.left, right))
        
    def test_output_dimensions(self):
        self.assertEqual(len(self.left)*len(self.right), 
                         len(cartesian_esqsum(self.left, self.right)))
        
class TestCartesianSqDiff(unittest.TestCase):
    
    def setUp(self):
        self.left = pd.DataFrame([[1, 1, 1], [5, 5, 5]])
        self.right = pd.DataFrame([[1, 1, 1], [4, 4, 4]])
        
    def test_cartesian_sqdiff_output(self):
        self.assertEqual(cartesian_sqdiff(self.left, self.right).tolist(), 
                         [[ 0, 0, 0], [9, 9, 9], [16, 16, 16], [1, 1, 1]])
    
    def test_wrong_first_input_type(self):
        left = np.array([[1, 1, 1], [5, 5, 5]])
        self.assertRaises(TypeError, cartesian_sqdiff, *(left, self.right))
        
    def test_wrong_second_input_type(self):
        right = np.array([[1, 1, 1], [4, 4, 4]])
        self.assertRaises(TypeError, cartesian_sqdiff, *(self.left, right))
    
    def test_wrong_first_input_dim(self):
        left = pd.DataFrame([[1, 5], [1, 5], [1, 5]])
        self.assertRaises(Exception, cartesian_sqdiff, *(left, self.right))
    
    def test_wrong_second_input_dim(self):
        right = pd.DataFrame([[1, 5], [1, 5], [1, 5]])
        self.assertRaises(Exception, cartesian_sqdiff, *(self.left, right))
        
    def test_output_dimensions(self):
        self.assertEqual(len(self.left)*len(self.right), 
                         len(cartesian_sqdiff(self.left, self.right)))
        
class TestCartesianDiff(unittest.TestCase):
    
    def setUp(self):
        self.left = pd.Series([0, 0, 0])
        self.right = pd.Series([1, 1, 1, 2, 2])
        
    def test_cartesian_diff_output(self):
        self.assertEqual(cartesian_diff(self.left, self.right).tolist(), 
                         [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2])
    
    def test_wrong_first_input_type(self):
        left = np.array([0, 0, 0])
        self.assertRaises(TypeError, cartesian_diff, *(left, self.right))
        
    def test_wrong_second_input_type(self):
        right = np.array([1, 1, 1, 2, 2])
        self.assertRaises(TypeError, cartesian_diff, *(self.left, right))
    
    def test_wrong_first_input_dim(self):
        left = pd.DataFrame([[0, 0, 0]])
        self.assertRaises(Exception, cartesian_diff, *(left, self.right))
    
    def test_wrong_second_input_dim(self):
        right = pd.DataFrame([[1, 1, 1, 2, 2]])
        self.assertRaises(Exception, cartesian_diff, *(self.left, right))
        
    def test_output_dimensions(self):
        self.assertEqual(len(self.left)*len(self.right), 
                         len(cartesian_diff(self.left, self.right)))
        
class TestFindLociDist(unittest.TestCase):
    
    def setUp(self):
        self.gene_dist = [0, 1000, 2000, 3000]
        self.nm_per_bp = 0.34
        self.pixel_dist = 100
    
    def test_find_loci_dist_output(self):        
        obs = find_loci_dist(self.gene_dist, self.nm_per_bp, self.pixel_dist).tolist()
        exp = [[0.0, 3.4, 6.8, 10.2], 
               [3.4, 0.0, 3.4, 6.8],
               [6.8, 3.4, 0.0, 3.4],
               [10.2, 6.8, 3.4, 0.0]]        
        self.assertTrue(np.allclose(obs, exp))
        
    def test_wrong_gene_dist_value_type(self):
        gene_dist = ['0', '1000', '2000', '3000']
        self.assertRaises(TypeError, find_loci_dist, *(gene_dist, self.nm_per_bp, self.pixel_dist))
    
    def test_wrong_nm_per_bp_input_type(self):
        nm_per_bp = '0.34'
        self.assertRaises(TypeError, find_loci_dist, *(self.gene_dist, nm_per_bp, self.pixel_dist))
    
    def test_neg_nm_per_bp_input(self):
        nm_per_bp = -0.3
        self.assertRaises(ValueError, find_loci_dist, *(self.gene_dist, nm_per_bp, self.pixel_dist))

    def test_wrong_pixel_dist_input_type(self):
        pixel_dist = '100'
        self.assertRaises(TypeError, find_loci_dist, *(self.gene_dist, self.nm_per_bp, pixel_dist))
    
    def test_neg_pixel_dist_input(self):
        pixel_dist = -100
        self.assertRaises(ValueError, find_loci_dist, *(self.gene_dist, self.nm_per_bp, pixel_dist))
    
    def test_output_shape(self):
        self.assertEqual(find_loci_dist(self.gene_dist, self.nm_per_bp, self.pixel_dist).shape, 
                         (len(self.gene_dist), len(self.gene_dist)))
        
    def test_output_symmetry(self):
        obs = find_loci_dist(self.gene_dist, self.nm_per_bp, self.pixel_dist)
        self.assertTrue(np.alltrue(obs==obs.T))
    
class TestCheckLpWgaps(unittest.TestCase):
    
    def setUp(self):
        self.left = pd.Series([0, 0, 0])
        self.right = pd.Series([1, 1, 1, 2, 2])
        self.l_p = .15
        self.loci_dist = np.array([[0, 1000, 2000],
                                   [1000, 0, 1000],
                                   [2000, 1000, 0]])
        
    def test_check_lp_wgaps_output(self):
        self.assertEqual(check_lp_wgaps(self.left, self.right, self.l_p, self.loci_dist).tolist(), 
                         [100., 100., 100., 200., 200., 
                          100., 100., 100., 200., 200.,
                          100., 100., 100., 200., 200.])
    
    def test_wrong_first_input_type(self):
        left = np.array([[0, 0, 0]])
        self.assertRaises(TypeError, check_lp_wgaps, *(left, self.right, self.l_p, self.loci_dist))
        
    def test_wrong_second_input_type(self):
        right = np.array([1, 1, 1, 2, 2])
        self.assertRaises(TypeError, check_lp_wgaps, *(self.left, right, self.l_p, self.loci_dist))
    
    def test_l_p_type(self):
        l_p = '.15'
        self.assertRaises(TypeError, check_lp_wgaps, *(self.left, self.right, l_p, self.loci_dist))
        
    def test_l_p_scale(self):
        l_p = 150 * 0.34
        self.assertRaises(ValueError, check_lp_wgaps, *(self.left, self.right, l_p, self.loci_dist))
        
    def test_loci_dist_dimensions(self):
        left = pd.Series([0, 0, 0])
        right = pd.Series([2, 2, 3, 3, 3])
        self.assertRaises(IndexError, check_lp_wgaps, *(left, right, self.l_p, self.loci_dist))
        
    def test_output_dimensions(self):
        self.assertEqual(len(self.left)*len(self.right), 
                         len(check_lp_wgaps(self.left, self.right, self.l_p, self.loci_dist)))
        
class TestLogBond(unittest.TestCase):
    
    def setUp(self):
        self.l_p = 3/(4*np.pi)
        self.ideal_l = 1
        self.r = 1
    
    def test_log_bond_output(self):        
        self.assertAlmostEqual(np.pi, log_bond(self.l_p, self.ideal_l, self.r))
    
    def test_log_bond_l_p_input_type(self):
        l_p = '3'
        self.assertRaises(TypeError, log_bond, *(l_p, self.ideal_l, self.r))
        
    def test_log_bond_l_p_input_value(self):
        l_p = -3/(4*np.pi)
        self.assertRaises(ValueError, log_bond, *(l_p, self.ideal_l, self.r))
        
    def test_log_bond_ideal_l_input_type(self):
        ideal_l = '1'
        self.assertRaises(TypeError, log_bond, *(self.l_p, ideal_l, self.r))
   
    def test_log_bond_ideal_l_input_value(self):
        ideal_l = -1
        self.assertRaises(ValueError, log_bond, *(self.l_p, ideal_l, self.r))
    
class TestLogBondVect(unittest.TestCase):
    
    def setUp(self):
        self.l_p_arr = np.full((2, 3), 3/(4*np.pi))
        self.ideal_l_arr = np.full((2, 3), 1)
        self.r_arr = np.full((2, 3), 1)
        
    def test_log_bond_vect_output(self):
        exp = np.full((2, 3), np.pi)
        obs = log_bond_vect(self.l_p_arr, self.ideal_l_arr, self.r_arr)
        self.assertTrue(np.allclose(exp, obs))
    
    def test_log_bond_vect_l_p_arr_type(self):
        l_p_arr = 3/(4*np.pi)
        self.assertRaises(TypeError, log_bond_vect, *(l_p_arr, self.ideal_l_arr, self.r_arr))
    
    def test_log_bond_vect_l_p_arr_value_type(self):
        l_p_arr = np.full((2, 3), '3')
        self.assertRaises(TypeError, log_bond_vect, *(l_p_arr, self.ideal_l_arr, self.r_arr))
        
    def test_log_bond_vect_l_p_arr_value(self):
        l_p_arr = np.full((2, 3), -3/(4*np.pi))
        self.assertRaises(ValueError, log_bond_vect, *(l_p_arr, self.ideal_l_arr, self.r_arr))        
    
    def test_log_bond_vect_ideal_l_arr_type(self):
        ideal_l_arr = 1
        self.assertRaises(TypeError, log_bond_vect, *(self.l_p_arr, ideal_l_arr, self.r_arr))
        
    def test_log_bond_vect_ideal_l_arr_value_type(self):
        ideal_l_arr = np.full((2, 3), '1')
        self.assertRaises(TypeError, log_bond_vect, *(self.l_p_arr, ideal_l_arr, self.r_arr))
    
    def test_log_bond_vect_ideal_l_arr_value(self):
        ideal_l_arr = np.full((2, 3), -1)
        self.assertRaises(ValueError, log_bond_vect, *(self.l_p_arr, ideal_l_arr, self.r_arr))
        
    def test_log_bond_vect_r_arr_value(self):
        r_arr = 1
        self.assertRaises(TypeError, log_bond_vect, *(self.l_p_arr, self.ideal_l_arr, r_arr))
        
    def test_log_bond_vect_r_arr_value_type(self):
        r_arr = np.full((2, 3), '1')
        self.assertRaises(TypeError, log_bond_vect, *(self.l_p_arr, self.ideal_l_arr, r_arr))
    
class TestCdfThresh(unittest.TestCase):
    
    def setUp(self):
        self.gene_dist = [0, 1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10]
        self.l_p_bp = 3/(4*np.pi)
        
    def test_cdf_thresh_base_calculation(self):
        with patch('random.shuffle', side_effect= lambda x: x) as mock_random:
            obs = cdf_thresh(self.gene_dist, self.l_p_bp)
            self.assertAlmostEqual(10*np.pi, obs)
        
    def test_cdf_thresh_check_gene_dist_type(self):
        gene_dist = 5000
        self.assertRaises(TypeError, cdf_thresh, *(gene_dist, self.l_p_bp))
        
    def test_cdf_thresh_check_gene_dist_value_type(self):
        gene_dist = ['0', '1', '2']
        self.assertRaises(TypeError, cdf_thresh, *(gene_dist, self.l_p_bp))
    
    def test_cdf_thresh_check_gene_dist_sorted(self):
        gene_dist = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.assertRaises(ValueError, cdf_thresh, *(gene_dist, self.l_p_bp))
        
    def test_cdf_thresh_check_gene_dist_positive_value(self):
        gene_dist = [-3, -2, -1]
        self.assertRaises(ValueError, cdf_thresh, *(gene_dist, self.l_p_bp))
        
    def test_cdf_thresh_check_l_p_bp_type(self):
        l_p_bp = '150'
        self.assertRaises(TypeError, cdf_thresh, *(self.gene_dist, l_p_bp))
        
    def test_cdf_thresh_check_l_p_bp_value(self):
        l_p_bp = -1
        self.assertRaises(ValueError, cdf_thresh, *(self.gene_dist, l_p_bp))
        
class TestEdgePenalty(unittest.TestCase):
    
    def setUp(self):
        self.skips = np.array([[0, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1]])
        self.l_p_bp = 3/(4*np.pi)
        self.corr_fac = 0.5
        self.bin_size = 1
        
    def test_edge_penalty_corr_fac_output(self):
        ones = np.ones(self.skips.shape)
        mult = -np.log( ((self.skips + 1)*(self.corr_fac**2))**(-3/2) * 
                        np.exp(-(self.skips+1)*np.pi) )
        sing = -np.log( ((ones)*(self.corr_fac**2))**(-3/2) * 
                        np.exp(-(ones)*np.pi) )
        ratio = mult/sing
        exp = np.divide(1.01*(self.skips+1), ratio)
        exp[self.skips == 0] = 1        
        obs = edge_penalty(self.skips, self.l_p_bp, self.corr_fac, self.bin_size)
        
        self.assertTrue(np.allclose(obs, exp))
    
    def test_edge_penalty_skips_input_type(self):
        cases = [([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]], self.l_p_bp, self.corr_fac, self.bin_size, TypeError),
                 ([[0., 0., 0., 1.1, 1.1], [0., 0., 0., 1.1, 1.1], [0., 0., 0., 1.1, 1.1]], self.l_p_bp, self.corr_fac, self.bin_size, TypeError)]
        
        for s, l, c, b, err in cases:
            with self.subTest(cases = cases):
                self.assertRaises(err, edge_penalty, *(s, l, c, b))
                
    def test_edge_penalty_l_p_bp_input_type(self):
        l_p_bp = '3'
        self.assertRaises(TypeError, edge_penalty, *(self.skips, l_p_bp, self.corr_fac, self.bin_size))
        
    def test_edge_penalty_l_p_bp_input_value(self):
        l_p_bp = -3/(4*np.pi)
        self.assertRaises(ValueError, edge_penalty, *(self.skips, l_p_bp, self.corr_fac, self.bin_size))
        
    def test_edge_penalty_corr_fac_input_type(self):
        corr_fac = '0.5'
        self.assertRaises(TypeError, edge_penalty, *(self.skips, self.l_p_bp, corr_fac, self.bin_size))
        
    def test_edge_penalty_corr_fac_input_value(self):
        corr_fac = -0.5
        self.assertRaises(ValueError, edge_penalty, *(self.skips, self.l_p_bp, corr_fac, self.bin_size))
        
    def test_edge_penalty_bin_size_input_type(self):
        bin_size = '1'
        self.assertRaises(TypeError, edge_penalty, *(self.skips, self.l_p_bp, self.corr_fac, bin_size))
        
    def test_edge_penalty_bin_size_input_value(self):
        bin_size = -1
        self.assertRaises(ValueError, edge_penalty, *(self.skips, self.l_p_bp, self.corr_fac, bin_size))
        
class TestEdgeWeights(unittest.TestCase):
    
    def setUp(self):
        self.bin_size = 1
        self.l_p_bp = 3/(4*np.pi)
        self.nm_per_bp = 1
        self.pixel_dist = 1
        self.theta = 0
        self.gene_dist = [0, 1, 2]
        self.loci_dist = find_loci_dist(self.gene_dist, self.nm_per_bp, self.pixel_dist)
        self.lim_min_dist = False        
    
    def test_edge_weights_xyz_base_calc(self):
        pts_clr_curr = pd.DataFrame({'x_hat':[0, 4],
                                     'y_hat':[0, 4],
                                     'z_hat':[0, 4],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[0, 0]})

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[1, 1]})
        
        obs = edge_weights(pts_clr_curr,
                           pts_clr_next, 
                           self.bin_size, 
                           self.l_p_bp,
                           self.nm_per_bp,
                           self.pixel_dist,
                           self.theta,
                           self.loci_dist, 
                           self.lim_min_dist).flatten()
        exp = np.array([3, 75, 27, 3]) * np.pi
        
        self.assertTrue(np.allclose(obs, exp))
        
    def test_edge_weights_sig_xyz_base_calc(self):
        pts_clr_curr = pd.DataFrame({'x_hat':[0, 4],
                                     'y_hat':[0, 4],
                                     'z_hat':[0, 4],
                                     'sig_x':[np.sqrt(1/(4*np.pi)), np.sqrt(1/(4*np.pi))],
                                     'sig_y':[np.sqrt(1/(4*np.pi)), np.sqrt(1/(4*np.pi))],
                                     'sig_z':[np.sqrt(1/(4*np.pi)), np.sqrt(1/(4*np.pi))],
                                     'hyb':[0, 0]})

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[np.sqrt(1/(4*np.pi)), np.sqrt(1/(4*np.pi))],
                                     'sig_y':[np.sqrt(1/(4*np.pi)), np.sqrt(1/(4*np.pi))],
                                     'sig_z':[np.sqrt(1/(4*np.pi)), np.sqrt(1/(4*np.pi))],
                                     'hyb':[1, 1]})
        
        obs = edge_weights(pts_clr_curr,
                           pts_clr_next, 
                           self.bin_size, 
                           self.l_p_bp,
                           self.nm_per_bp,
                           self.pixel_dist,
                           self.theta,
                           self.loci_dist, 
                           self.lim_min_dist).flatten()
        
        exp = -np.log(2**(-3/2)) + np.array([3, 75, 27, 3])*np.pi/2 
        
        self.assertTrue(np.allclose(obs, exp))
        
    def test_edge_weights_bond_gap_base_calc(self):
        pts_clr_curr = pd.DataFrame({'x_hat':[0, 4],
                                     'y_hat':[0, 4],
                                     'z_hat':[0, 4],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[0, 0]})

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[2, 2]})

        obs = edge_weights(pts_clr_curr,
                           pts_clr_next, 
                           self.bin_size, 
                           self.l_p_bp,
                           self.nm_per_bp,
                           self.pixel_dist,
                           self.theta,
                           self.loci_dist, 
                           self.lim_min_dist).flatten()
        exp = -np.log(2**(-3/2)) + np.array([3, 75, 27, 3])*np.pi/2 
        
        self.assertTrue(np.allclose(obs, exp))
        
    def test_edge_weights_input_dataframe_type(self):
        pts_clr_curr = {'x_hat':[0, 4],
                        'y_hat':[0, 4],
                        'z_hat':[0, 4],
                        'sig_x':[0, 0],
                        'sig_y':[0, 0],
                        'sig_z':[0, 0],
                        'hyb':[0, 0]}

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[2, 2]})
        
        self.assertRaises(TypeError, 
                          edge_weights, 
                          *(pts_clr_curr,
                            pts_clr_next,
                            self.bin_size, 
                            self.l_p_bp,
                            self.nm_per_bp,
                            self.pixel_dist,
                            self.theta,
                            self.loci_dist, 
                            self.lim_min_dist) )
        
    def test_edge_weights_input_dataframe_columns(self):
        pts_clr_curr = pd.DataFrame({'x':[0, 4],
                                     'y':[0, 4],
                                     'z':[0, 4],
                                     'sx':[0, 0],
                                     'sy':[0, 0],
                                     'sz':[0, 0],
                                     'h':[0, 0]})

        pts_clr_next = pd.DataFrame({'x':[1, 5],
                                     'y':[1, 5],
                                     'z':[1, 5],
                                     'sx':[0, 0],
                                     'sy':[0, 0],
                                     'sz':[0, 0],
                                     'h':[2, 2]})
        
        self.assertRaises(KeyError, 
                          edge_weights, 
                          *(pts_clr_curr,
                            pts_clr_next,
                            self.bin_size, 
                            self.l_p_bp,
                            self.nm_per_bp,
                            self.pixel_dist,
                            self.theta,
                            self.loci_dist, 
                            self.lim_min_dist) )
        
    def test_edge_weights_bond_gap_within_limit(self):
        pts_clr_curr = pd.DataFrame({'x_hat':[0, 4],
                                     'y_hat':[0, 4],
                                     'z_hat':[0, 4],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[0, 0]})

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[3, 3]})
        
        self.assertRaises(IndexError, 
                          edge_weights, 
                          *(pts_clr_curr,
                            pts_clr_next,
                            self.bin_size, 
                            self.l_p_bp,
                            self.nm_per_bp,
                            self.pixel_dist,
                            self.theta,
                            self.loci_dist, 
                            self.lim_min_dist) )
        
    def test_edge_weights_hyb_sorted(self):
        pts_clr_curr = pd.DataFrame({'x_hat':[0, 4],
                                     'y_hat':[0, 4],
                                     'z_hat':[0, 4],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[0, 0]})

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[2, 1]})
        
        self.assertRaises(ValueError, 
                          edge_weights, 
                          *(pts_clr_curr,
                            pts_clr_next,
                            self.bin_size, 
                            self.l_p_bp,
                            self.nm_per_bp,
                            self.pixel_dist,
                            self.theta,
                            self.loci_dist, 
                            self.lim_min_dist) )
        
    def test_edge_weights_params_positive_number(self):
        cases = [('1', self.l_p_bp, self.nm_per_bp, self.pixel_dist, self.theta),
                 (-1, self.l_p_bp, self.nm_per_bp, self.pixel_dist, self.theta),
                 (self.bin_size, '3', self.nm_per_bp, self.pixel_dist, self.theta),
                 (self.bin_size, -3/(4*np.pi), self.nm_per_bp, self.pixel_dist, self.theta),
                 (self.bin_size, self.l_p_bp, '1', self.pixel_dist, self.theta),
                 (self.bin_size, self.l_p_bp, -1, self.pixel_dist, self.theta),
                 (self.bin_size, self.l_p_bp, self.nm_per_bp, '1', self.theta),
                 (self.bin_size, self.l_p_bp, self.nm_per_bp, -1, self.theta),
                 (self.bin_size, self.l_p_bp, self.nm_per_bp, self.pixel_dist, '0'),
                 (self.bin_size, self.l_p_bp, self.nm_per_bp, self.pixel_dist, -0.5)]
        
        pts_clr_curr = pd.DataFrame({'x_hat':[0, 4],
                                     'y_hat':[0, 4],
                                     'z_hat':[0, 4],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[0, 0]})

        pts_clr_next = pd.DataFrame({'x_hat':[1, 5],
                                     'y_hat':[1, 5],
                                     'z_hat':[1, 5],
                                     'sig_x':[0, 0],
                                     'sig_y':[0, 0],
                                     'sig_z':[0, 0],
                                     'hyb':[1, 1]})
        for b, l, n, p, t in cases:
            with self.subTest(cases = cases):
                self.assertRaises(ValueError, 
                                  edge_weights, 
                                  *(pts_clr_curr, pts_clr_next, b, l, n, p, t, self.loci_dist, self.lim_min_dist))
        
        
class TestBoundaryInit(unittest.TestCase):
    
    def setUp(self):
        self.gene_dist = [0, 1, 2, 3, 4, 5]
        self.l_p_bp = 3/(4*np.pi)
        self.nm_per_bp = 1
        self.pixel_dist = 1
        self.corr_fac = self.nm_per_bp / self.pixel_dist
        self.n_colours = len(self.gene_dist)
        self.exp_stretch = 1
        self.stretch_factor = 1
        self.lim_init_skip = True
        self.init_skip = 2
        self.end_skip = 4
        self.loci_dist = find_loci_dist(self.gene_dist, self.nm_per_bp, self.pixel_dist)
        
    def test_boundary_init_base_calc(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        lim_init_skip = False
        
        obs = boundary_init(trans_mat,
                            self.loci_dist,
                            self.l_p_bp,
                            self.corr_fac,
                            self.n_colours,
                            cell_pts,
                            self.exp_stretch,
                            self.stretch_factor,
                            lim_init_skip,
                            self.init_skip,
                            self.end_skip)
        
        obs_row = obs[0, :].flatten()
        obs_col = obs[:, -1].flatten()
        
        exp_row = np.concatenate( ([0,], cell_pts['hyb'].values, [0,]) ) * np.pi
        exp_col = np.concatenate( ([0,], max(cell_pts['hyb']) - cell_pts['hyb'].values,[0,]))* np.pi
        
        self.assertTrue(np.allclose(obs_row, exp_row) and np.allclose(obs_col, exp_col))
        
    def test_boundary_init_lim_init_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        lim_init_skip = True
        
        obs = boundary_init(trans_mat,
                            self.loci_dist,
                            self.l_p_bp,
                            self.corr_fac,
                            self.n_colours,
                            cell_pts,
                            self.exp_stretch,
                            self.stretch_factor,
                            lim_init_skip,
                            self.init_skip,
                            self.end_skip)
        
        obs_row = obs[0, :].flatten()
        obs_col = obs[:, -1].flatten()
        
        exp_row = np.array([0, 0, 1, 2, 0, 0, 0, 0]) * np.pi 
        exp_col = np.array([0, 0, 0, 0, 0, 1, 0, 0]) * np.pi
        
        self.assertTrue(np.allclose(obs_row, exp_row) and np.allclose(obs_col, exp_col))
        
    def test_boundary_init_mult_cand_of_same_hyb_have_same_weight(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                                 'y_hat':[0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                                 'z_hat':[0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        lim_init_skip = False
        
        obs = boundary_init(trans_mat,
                            self.loci_dist,
                            self.l_p_bp,
                            self.corr_fac,
                            self.n_colours,
                            cell_pts,
                            self.exp_stretch,
                            self.stretch_factor,
                            lim_init_skip,
                            self.init_skip,
                            self.end_skip)
        
        obs_row = obs[0, :].flatten()
        obs_col = obs[:, -1].flatten()
        
        exp_row = np.concatenate( ([0,], cell_pts['hyb'].values, [0,]) ) * np.pi
        exp_col = np.concatenate( ([0,], max(cell_pts['hyb']) - cell_pts['hyb'].values,[0,]))* np.pi
        
        self.assertTrue(np.allclose(obs_row, exp_row) and np.allclose(obs_col, exp_col))
        
    def test_boundary_init_corr_fac(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        corr_fac = 1/2
        lim_init_skip = False
        
        obs = boundary_init(trans_mat,
                            self.loci_dist,
                            self.l_p_bp,
                            corr_fac,
                            self.n_colours,
                            cell_pts,
                            self.exp_stretch,
                            self.stretch_factor,
                            lim_init_skip,
                            self.init_skip,
                            self.end_skip)
        
        obs_row = obs[0, :].flatten()
        obs_col = obs[:, -1].flatten()
        
        exp_row = np.concatenate( ([0,], cell_pts['hyb'].values, [0,]) ) * (np.pi/corr_fac-np.log(1/(corr_fac**1.5)))
        exp_col = np.concatenate( ([0,], max(cell_pts['hyb']) - cell_pts['hyb'].values,[0,])) * (np.pi/corr_fac-np.log(1/(corr_fac**1.5)))
        
        self.assertTrue(np.allclose(obs_row, exp_row) and np.allclose(obs_col, exp_col))
        
    def test_boundary_init_stretch_factor(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        corr_fac = 1
        stretch_factor = 3
        lim_init_skip = False
        
        obs = boundary_init(trans_mat,
                            self.loci_dist,
                            self.l_p_bp,
                            corr_fac,
                            self.n_colours,
                            cell_pts,
                            self.exp_stretch,
                            stretch_factor,
                            lim_init_skip,
                            self.init_skip,
                            self.end_skip)
        
        obs_row = obs[0, :].flatten()
        obs_col = obs[:, -1].flatten()
        
        exp_row = np.concatenate( ([0,], cell_pts['hyb'].values, [0,]) ) * ( (stretch_factor**2) *np.pi/corr_fac-np.log(1/(corr_fac**1.5)))
        exp_col = np.concatenate( ([0,], max(cell_pts['hyb']) - cell_pts['hyb'].values,[0,])) * ( (stretch_factor**2) *np.pi/corr_fac-np.log(1/(corr_fac**1.5)))
        
        self.assertTrue(np.allclose(obs_row, exp_row) and np.allclose(obs_col, exp_col))
        
    def test_boundary_init_exp_stretch(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        corr_fac = 1
        stretch_factor = 3
        exp_stretch = 3
        lim_init_skip = False
        
        obs = boundary_init(trans_mat,
                            self.loci_dist,
                            self.l_p_bp,
                            corr_fac,
                            self.n_colours,
                            cell_pts,
                            exp_stretch,
                            stretch_factor,
                            lim_init_skip,
                            self.init_skip,
                            self.end_skip)
        
        obs_row = obs[0, :].flatten()
        obs_col = obs[:, -1].flatten()
        
        exp_row = np.concatenate( ([0,], cell_pts['hyb'].values, [0,]) ) * \
                  ( (stretch_factor**2)/exp_stretch*np.pi/corr_fac-np.log(1/((exp_stretch*corr_fac)**1.5)))
        exp_col = np.concatenate( ([0,], max(cell_pts['hyb']) - cell_pts['hyb'].values,[0,])) * \
                  ( (stretch_factor**2)/exp_stretch*np.pi/corr_fac-np.log(1/((exp_stretch*corr_fac)**1.5)))
        
        self.assertTrue(np.allclose(obs_row, exp_row) and np.allclose(obs_col, exp_col))
        
    def test_boundary_init_cell_pts_input_type(self):
        cell_pts = {'x_hat':[0, 1, 2, 3, 4, 5],
                    'y_hat':[0, 1, 2, 3, 4, 5],
                    'z_hat':[0, 1, 2, 3, 4, 5],
                    'sig_x':[0, 0, 0, 0, 0, 0],
                    'sig_y':[0, 0, 0, 0, 0, 0],
                    'sig_z':[0, 0, 0, 0, 0, 0],
                    'hyb':[0, 1, 2, 3, 4, 5]}
        
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        lim_init_skip = False
        
        self.assertRaises(TypeError, boundary_init, *(trans_mat,
                                                      self.loci_dist,
                                                      self.l_p_bp,
                                                      self.corr_fac,
                                                      self.n_colours,
                                                      cell_pts,
                                                      self.exp_stretch,
                                                      self.stretch_factor,
                                                      lim_init_skip,
                                                      self.init_skip,
                                                      self.end_skip))
        
    def test_boundary_init_cell_pts_missing_columns(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        lim_init_skip = False
        
        self.assertRaises(KeyError, boundary_init, *(trans_mat,
                                                     self.loci_dist,
                                                     self.l_p_bp,
                                                     self.corr_fac,
                                                     self.n_colours,
                                                     cell_pts,
                                                     self.exp_stretch,
                                                     self.stretch_factor,
                                                     lim_init_skip,
                                                     self.init_skip,
                                                     self.end_skip))
        
    def test_boundary_init_cell_pts_unsorted_hyb(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 3, 2, 4, 5],
                                 'y_hat':[0, 1, 3, 2, 4, 5],
                                 'z_hat':[0, 1, 3, 2, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 3, 2, 4, 5]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        lim_init_skip = False
        
        self.assertRaises(IndexError, boundary_init, *(trans_mat,
                                                     self.loci_dist,
                                                     self.l_p_bp,
                                                     self.corr_fac,
                                                     self.n_colours,
                                                     cell_pts,
                                                     self.exp_stretch,
                                                     self.stretch_factor,
                                                     lim_init_skip,
                                                     self.init_skip,
                                                     self.end_skip))
        
    def test_boundary_init_wrong_trans_mat_type(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = [[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]]
        lim_init_skip = False
        
        self.assertRaises(TypeError, boundary_init, *(trans_mat,
                                                     self.loci_dist,
                                                     self.l_p_bp,
                                                     self.corr_fac,
                                                     self.n_colours,
                                                     cell_pts,
                                                     self.exp_stretch,
                                                     self.stretch_factor,
                                                     lim_init_skip,
                                                     self.init_skip,
                                                     self.end_skip))
        
    def test_boundary_init_wrong_loci_dist_type(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        loci_dist = [[0., 1., 2., 3., 4., 5.],
                     [1., 0., 1., 2., 3., 4.],
                     [2., 1., 0., 1., 2., 3.],
                     [3., 2., 1., 0., 1., 2.],
                     [4., 3., 2., 1., 0., 1.],
                     [5., 4., 3., 2., 1., 0.]]
        lim_init_skip = False
        
        self.assertRaises(TypeError, boundary_init, *(trans_mat,
                                                     loci_dist,
                                                     self.l_p_bp,
                                                     self.corr_fac,
                                                     self.n_colours,
                                                     cell_pts,
                                                     self.exp_stretch,
                                                     self.stretch_factor,
                                                     lim_init_skip,
                                                     self.init_skip,
                                                     self.end_skip))
        
    def test_boundary_init_wrong_loci_dist_dim(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        loci_dist = np.array([[0., 1., 2., 3., 4., 5.],
                              [1., 0., 1., 2., 3., 4.],
                              [2., 1., 0., 1., 2., 3.],
                              [3., 2., 1., 0., 1., 2.],
                              [5., 4., 3., 2., 1., 0.]])
        lim_init_skip = False
        
        self.assertRaises(ValueError, boundary_init, *(trans_mat,
                                                     loci_dist,
                                                     self.l_p_bp,
                                                     self.corr_fac,
                                                     self.n_colours,
                                                     cell_pts,
                                                     self.exp_stretch,
                                                     self.stretch_factor,
                                                     lim_init_skip,
                                                     self.init_skip,
                                                     self.end_skip))
        
    def test_boundary_init_wrong_input_values(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        lim_init_skip = False
        
        cases = [(TypeError, '3', self.corr_fac, self.exp_stretch, self.stretch_factor),
                 (ValueError, -3/(4*np.pi), self.corr_fac, self.exp_stretch, self.stretch_factor),
                 (TypeError,self.l_p_bp, '1', self.exp_stretch, self.stretch_factor),
                 (ValueError,self.l_p_bp, -1, self.exp_stretch, self.stretch_factor),
                 (TypeError,self.l_p_bp, self.corr_fac, '1', self.stretch_factor),
                 (ValueError,self.l_p_bp, self.corr_fac, -1, self.stretch_factor),
                 (TypeError,self.l_p_bp, self.corr_fac, self.exp_stretch, '1'),
                 (ValueError,self.l_p_bp, self.corr_fac, self.exp_stretch, -1)]
        
        for err, l_p_bp, corr_fac, exp_stretch, stretch_factor in cases:
            with self.subTest(cases = cases):
                self.assertRaises(err, boundary_init, *(trans_mat,
                                                               self.loci_dist,
                                                               l_p_bp,
                                                               corr_fac,
                                                               self.n_colours,
                                                               cell_pts,
                                                               exp_stretch,
                                                               stretch_factor,
                                                               lim_init_skip,
                                                               self.init_skip,
                                                               self.end_skip))
        
    def test_boundary_init_wrong_init_end_skip_values(self):
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
                                 
        cell_pts['CurrIndex'] = cell_pts.index
        trans_mat = np.zeros((len(self.gene_dist), len(self.gene_dist)))
        lim_init_skip = False
        
        cases = [(TypeError, 2.1, 3),
                 (ValueError, -1, 3),
                 (ValueError, 6, 6),
                 (TypeError, 2, 3.1),
                 (ValueError, 0, -1),
                 (ValueError, 3, 2)]
        
        for err, init_skip, end_skip in cases:
            with self.subTest(cases = cases):
                self.assertRaises(err, boundary_init, *(trans_mat,
                                                        self.loci_dist,
                                                        self.l_p_bp,
                                                        self.corr_fac,
                                                        self.n_colours,
                                                        cell_pts,
                                                        self.exp_stretch,
                                                        self.stretch_factor,
                                                        lim_init_skip,
                                                        init_skip,
                                                        end_skip))
                
        
    def test_boundary_init_loci_dist_n_colours_shape(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 4, 5],
                                 'y_hat':[0, 1, 2, 3, 4, 5],
                                 'z_hat':[0, 1, 2, 3, 4, 5],
                                 'sig_x':[0, 0, 0, 0, 0, 0],
                                 'sig_y':[0, 0, 0, 0, 0, 0],
                                 'sig_z':[0, 0, 0, 0, 0, 0],
                                 'hyb':[0, 1, 2, 3, 4, 5]})
        cell_pts['CurrIndex'] = cell_pts.index        
        loci_dist = find_loci_dist(self.gene_dist,
                                   self.nm_per_bp,
                                   self.pixel_dist)
        
        trans_mat = np.zeros((cell_pts.shape[0], cell_pts.shape[0]))
        
        cases = [(np.expand_dims(loci_dist[0, :], 0), 6, trans_mat),
                 (np.expand_dims(loci_dist[:, 0], 0), 6, trans_mat),
                 (loci_dist, 5, trans_mat),]
        
        for loci_dist, n_colours, trans_mat in cases:
            with self.subTest(cases = cases):
                self.assertRaises(ValueError, 
                                  boundary_init, 
                                  *(trans_mat, 
                                    loci_dist,
                                    self.l_p_bp,
                                    self.corr_fac,
                                    n_colours,
                                    cell_pts,
                                    self.exp_stretch,
                                    self.stretch_factor,
                                    self.lim_init_skip,
                                    self.init_skip,
                                    self.end_skip))

class TestFindChr(unittest.TestCase):
    
    def setUp(self):
        self.gene_dist = [i for i in range(10)]
        self.bin_size = 1
        self.nm_per_bp = 1
        self.pixel_dist = 1
        self.l_p_bp = 1e-1
        self.n_colours = len(self.gene_dist)
        self.exp_stretch = 1
        self.stretch_factor = 1.2
        self.num_skip = 4
        self.total_num_skip_frac = 0.7
        self.init_skip_frac = 0.3
        self.theta = 0
        self.norm_skip_penalty = False
        self.lim_init_skip = False
        self.lim_min_dist = False

    def test_find_chr_base_calc(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(10)],
                         'y_hat':[0 for i in range(10)],
                         'z_hat':[0 for i in range(10)],
                         'sig_x':[0 for i in range(10)],
                         'sig_y':[0 for i in range(10)],
                         'sig_z':[0 for i in range(10)],
                         'hyb'  :[i for i in range(10)]})
        
        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              self.init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              self.lim_init_skip,
                              self.lim_min_dist)

        self.assertTrue(np.all(np.equal(path, [i for i in range(10)])))

    def test_find_chr_stretch_factor_greater_than_exp_stretch(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(10)],
                         'y_hat':[0 for i in range(10)],
                         'z_hat':[0 for i in range(10)],
                         'sig_x':[0 for i in range(10)],
                         'sig_y':[0 for i in range(10)],
                         'sig_z':[0 for i in range(10)],
                         'hyb'  :[i for i in range(10)]})
        
        stretch_factor = 1

        self.assertRaises(ValueError,
                          find_chr,
                          *(cell_pts,
                            self.gene_dist,
                            self.bin_size, 
                            self.nm_per_bp,
                            self.pixel_dist,
                            self.l_p_bp,
                            stretch_factor,
                            self.exp_stretch,
                            self.num_skip,
                            self.total_num_skip_frac,
                            self.init_skip_frac,
                            self.theta,
                            self.norm_skip_penalty,
                            self.lim_init_skip,
                            self.lim_min_dist))
        
    def test_find_chr_contour_length_scale_relative_to_persistence_length(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(10)],
                         'y_hat':[0 for i in range(10)],
                         'z_hat':[0 for i in range(10)],
                         'sig_x':[0 for i in range(10)],
                         'sig_y':[0 for i in range(10)],
                         'sig_z':[0 for i in range(10)],
                         'hyb'  :[i for i in range(10)]})
        
        l_p_bp = 1

        self.assertRaises(ValueError,
                          find_chr,
                          *(cell_pts,
                            self.gene_dist,
                            self.bin_size, 
                            self.nm_per_bp,
                            self.pixel_dist,
                            l_p_bp,
                            self.stretch_factor,
                            self.exp_stretch,
                            self.num_skip,
                            self.total_num_skip_frac,
                            self.init_skip_frac,
                            self.theta,
                            self.norm_skip_penalty,
                            self.lim_init_skip,
                            self.lim_min_dist))

    def test_find_chr_contour_length_scale_relative_to_persistence_length(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(10)],
                         'y_hat':[0 for i in range(10)],
                         'z_hat':[0 for i in range(10)],
                         'sig_x':[0 for i in range(10)],
                         'sig_y':[0 for i in range(10)],
                         'sig_z':[0 for i in range(10)],
                         'hyb'  :[i for i in range(10)]})
        
        l_p_bp = 1

        self.assertRaises(ValueError,
                          find_chr,
                          *(cell_pts,
                            self.gene_dist,
                            self.bin_size, 
                            self.nm_per_bp,
                            self.pixel_dist,
                            l_p_bp,
                            self.stretch_factor,
                            self.exp_stretch,
                            self.num_skip,
                            self.total_num_skip_frac,
                            self.init_skip_frac,
                            self.theta,
                            self.norm_skip_penalty,
                            self.lim_init_skip,
                            self.lim_min_dist))
        
    def test_find_chr_contour_length_gene_dist_scale_relative_to_persistence_length(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(10)],
                         'y_hat':[0 for i in range(10)],
                         'z_hat':[0 for i in range(10)],
                         'sig_x':[0 for i in range(10)],
                         'sig_y':[0 for i in range(10)],
                         'sig_z':[0 for i in range(10)],
                         'hyb'  :[i for i in range(10)]})
        
        bin_size = 10
        nm_per_bp = 1e-3
        l_p_bp = 1
        self.assertRaises(ValueError,
                          find_chr,
                          *(cell_pts,
                            self.gene_dist,
                            bin_size, 
                            nm_per_bp,
                            self.pixel_dist,
                            l_p_bp,
                            self.stretch_factor,
                            self.exp_stretch,
                            self.num_skip,
                            self.total_num_skip_frac,
                            self.init_skip_frac,
                            self.theta,
                            self.norm_skip_penalty,
                            self.lim_init_skip,
                            self.lim_min_dist))
        
    def test_find_chr_prevent_init_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(3, 10)],
                                 'y_hat':[0 for i in range(3, 10)],
                                 'z_hat':[0 for i in range(3, 10)],
                                 'sig_x':[0 for i in range(3, 10)],
                                 'sig_y':[0 for i in range(3, 10)],
                                 'sig_z':[0 for i in range(3, 10)],
                                 'hyb'  :[i for i in range(3, 10)]})
        
        
        lim_init_skip = True
        init_skip_frac = 0.2

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [])
        
    def test_find_chr_allow_init_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(3, 10)],
                                 'y_hat':[0 for i in range(3, 10)],
                                 'z_hat':[0 for i in range(3, 10)],
                                 'sig_x':[0 for i in range(3, 10)],
                                 'sig_y':[0 for i in range(3, 10)],
                                 'sig_z':[0 for i in range(3, 10)],
                                 'hyb'  :[i for i in range(3, 10)]})
        
        
        lim_init_skip = True
        init_skip_frac = 0.3

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [0, 1, 2, 3, 4, 5, 6])
        
    def test_find_chr_prevent_end_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(0, 8)],
                                 'y_hat':[0 for i in range(0, 8)],
                                 'z_hat':[0 for i in range(0, 8)],
                                 'sig_x':[0 for i in range(0, 8)],
                                 'sig_y':[0 for i in range(0, 8)],
                                 'sig_z':[0 for i in range(0, 8)],
                                 'hyb'  :[i for i in range(0, 8)]})
        
        
        lim_init_skip = True
        init_skip_frac = 0.2

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [])
        
    def test_find_chr_allow_end_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[i for i in range(0, 8)],
                                 'y_hat':[0 for i in range(0, 8)],
                                 'z_hat':[0 for i in range(0, 8)],
                                 'sig_x':[0 for i in range(0, 8)],
                                 'sig_y':[0 for i in range(0, 8)],
                                 'sig_z':[0 for i in range(0, 8)],
                                 'hyb'  :[i for i in range(0, 8)]})
        
        lim_init_skip = True
        init_skip_frac = 0.3

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [0, 1, 2, 3, 4, 5, 6, 7])
        
    def test_find_chr_prevent_middle_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 5, 6, 7, 8, 9],
                                 'y_hat':[0 for i in range(0, 8)],
                                 'z_hat':[0 for i in range(0, 8)],
                                 'sig_x':[0 for i in range(0, 8)],
                                 'sig_y':[0 for i in range(0, 8)],
                                 'sig_z':[0 for i in range(0, 8)],
                                 'hyb'  :[0, 1, 2, 5, 6, 7, 8, 9]})
        
        lim_init_skip = True
        num_skip = 3

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              num_skip,
                              self.total_num_skip_frac,
                              self.init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [])
        
    def test_find_chr_allow_middle_skip(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 5, 6, 7, 8, 9],
                                 'y_hat':[0 for i in range(0, 8)],
                                 'z_hat':[0 for i in range(0, 8)],
                                 'sig_x':[0 for i in range(0, 8)],
                                 'sig_y':[0 for i in range(0, 8)],
                                 'sig_z':[0 for i in range(0, 8)],
                                 'hyb'  :[0, 1, 2, 5, 6, 7, 8, 9]})
        
        lim_init_skip = True
        num_skip = 4

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              num_skip,
                              self.total_num_skip_frac,
                              self.init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [0, 1, 2, 3, 4, 5, 6, 7])
        
    def test_find_chr_avoid_false_positive(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 7, 4, 5, 6, 7, 8, 9],
                                 'y_hat':[0 for i in range(11)],
                                 'z_hat':[0 for i in range(11)],
                                 'sig_x':[0 for i in range(11)],
                                 'sig_y':[0 for i in range(11)],
                                 'sig_z':[0 for i in range(11)],
                                 'hyb'  :[0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9]})
        
        lim_init_skip = True

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              self.init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10])

    def test_find_chr_avoid_false_positive_with_false_negative(self):
        
        cell_pts = pd.DataFrame({'x_hat':[0, 1, 2, 3, 7, 5, 6, 7, 8, 9],
                                 'y_hat':[0 for i in range(10)],
                                 'z_hat':[0 for i in range(10)],
                                 'sig_x':[0 for i in range(10)],
                                 'sig_y':[0 for i in range(10)],
                                 'sig_z':[0 for i in range(10)],
                                 'hyb'  :[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})

        lim_init_skip = True

        _, path, _ = find_chr(cell_pts,
                              self.gene_dist,
                              self.bin_size, 
                              self.nm_per_bp,
                              self.pixel_dist,
                              self.l_p_bp,
                              self.stretch_factor,
                              self.exp_stretch,
                              self.num_skip,
                              self.total_num_skip_frac,
                              self.init_skip_frac,
                              self.theta,
                              self.norm_skip_penalty,
                              lim_init_skip,
                              self.lim_min_dist)
        
        self.assertEqual(path, [0, 1, 2, 3, 5, 6, 7, 8, 9])

    def test_find_chr_polymer_base_calc(self):
        
        cell_pts = pd.read_csv('../test/wlc-100hyb-10000kb-1chr-1true-0fp-0fn.csv',
                       index_col=0)
        
        bin_size = 10000
        gene_dist = [i*bin_size for i in range(100)]
        lim_init_skip = True

        _, path, _ = find_chr(cell_pts,
                              gene_dist = gene_dist,
                              bin_size = 10000, 
                              nm_per_bp = 0.34,
                              pixel_dist = 100,
                              l_p_bp = 150.,
                              stretch_factor=1.2,
                              exp_stretch=1,
                              num_skip=5,
                              total_num_skip_frac=0.7,
                              init_skip_frac=0.3,
                              theta=0,
                              norm_skip_penalty=True,
                              lim_init_skip=True,
                              lim_min_dist=True)
        
        self.assertEqual(path, [i for i in range(100)])
        
    def test_find_chr_polymer_avoid_false_positive(self):
        
        cell_pts = pd.read_csv('../test/wlc-100hyb-10000kb-1chr-1true-010fp-0fn.csv',
                       index_col=0)
        cell_pts.sort_values(by='hyb', inplace = True)
        cell_pts.reset_index(drop=True, inplace = True)
        bin_size = 10000
        gene_dist = [i*bin_size for i in range(100)]
        lim_init_skip = True
        fp_indeces = [i for i in cell_pts[cell_pts['FP']==True].index]

        _, path, _ = find_chr(cell_pts,
                              gene_dist = gene_dist,
                              bin_size = 10000, 
                              nm_per_bp = 0.34,
                              pixel_dist = 100,
                              l_p_bp = 150.,
                              stretch_factor=1.2,
                              exp_stretch=1,
                              num_skip=5,
                              total_num_skip_frac=0.7,
                              init_skip_frac=0.3,
                              theta=0,
                              norm_skip_penalty=True,
                              lim_init_skip=True,
                              lim_min_dist=True)
        
        self.assertTrue(elem not in path for elem in fp_indeces)
        
    def test_find_chr_polymer_skip_false_negative(self):
        
        cell_pts = pd.read_csv('../test/wlc-100hyb-10000kb-1chr-1true-0fp-010fn.csv',
                               index_col=0)
        fn_hyb = [i for i in cell_pts[cell_pts['FN']==True].hyb]
        cell_pts = cell_pts[cell_pts['FN'] != True]
        cell_pts.reset_index(drop=True, inplace = True)
        bin_size = 10000
        gene_dist = [i*bin_size for i in range(100)]
        lim_init_skip = True
        

        _, path, _ = find_chr(cell_pts,
                              gene_dist = gene_dist,
                              bin_size = 10000, 
                              nm_per_bp = 0.34,
                              pixel_dist = 100,
                              l_p_bp = 150.,
                              stretch_factor=1.2,
                              exp_stretch=1,
                              num_skip=5,
                              total_num_skip_frac=0.7,
                              init_skip_frac=0.3,
                              theta=0,
                              norm_skip_penalty=True,
                              lim_init_skip=True,
                              lim_min_dist=True)
        
        self.assertTrue(elem not in cell_pts.iloc[path]['hyb'] for elem in fn_hyb)

    def test_find_chr_polymer_avoid_fp_amid_false_neg(self):
        
        cell_pts = pd.read_csv('../test/wlc-100hyb-10000kb-1chr-1true-010fp-0fn-trial2.csv',
                               index_col=0)
        cell_pts.sort_values(by='hyb', inplace = True)
        cell_pts.reset_index(inplace = True, drop=True)
        fp_hyb = cell_pts[cell_pts['FP'] == True].hyb
        idx_exclude = cell_pts[(cell_pts['hyb'].isin(fp_hyb)) & (cell_pts['FP'] == False)].index
        cell_pts = copy.deepcopy(cell_pts[~cell_pts.index.isin(idx_exclude)])
        bin_size = 10000
        gene_dist = [i*bin_size for i in range(100)]
        lim_init_skip = True        

        _, path, _ = find_chr(cell_pts,
                              gene_dist = gene_dist,
                              bin_size = 10000, 
                              nm_per_bp = 0.34,
                              pixel_dist = 100,
                              l_p_bp = 150.,
                              stretch_factor=1.2,
                              exp_stretch=1,
                              num_skip=5,
                              total_num_skip_frac=0.7,
                              init_skip_frac=0.3,
                              theta=0,
                              norm_skip_penalty=True,
                              lim_init_skip=True,
                              lim_min_dist=True)
        
        self.assertTrue(sum(cell_pts.iloc[path]['FP']) < 10)
        

class TestFindAllChr(unittest.TestCase):
    
    def setUp(self):
        self.bin_size = 10000
        self.gene_dist = [i*self.bin_size for i in range(100)]
        self.nm_per_bp = 0.34
        self.pixel_dist = 100
        self.l_p_bp = 150
        self.n_colours = len(self.gene_dist)
        self.exp_stretch = 1
        self.stretch_factor = 1.2
        self.num_skip = 7
        self.total_num_skip_frac = 0.7
        self.init_skip_frac = 0.3
        self.theta = 0
        self.norm_skip_penalty = True
        self.lim_init_skip = True
        self.lim_min_dist = True

    def test_find_all_chr_diff_cdf_thresholds(self):
        
        cell_pts = pd.read_csv('../test/wlc-100hyb-10000kb-4chr-4true-010fp-010fn.csv')
        cases = [(902, 1), (904, 2), (961, 3), (970, 4)]
        
        for thresh, num_chr in cases:
            with self.subTest(cases = cases):
                with patch('jie.aligner.cdf_thresh', return_value = thresh) as _:

                    res = find_all_chr(cell_pts, 
                                       gene_dist=self.gene_dist, 
                                       bin_size = self.bin_size, 
                                       nm_per_bp = self.nm_per_bp, 
                                       pixel_dist = self.pixel_dist, 
                                       l_p_bp = self.l_p_bp,
                                       num_skip = self.num_skip,
                                       stretch_factor = self.stretch_factor,
                                       init_skip_frac = self.init_skip_frac,
                                       norm_skip_penalty = self.norm_skip_penalty,
                                       lim_init_skip = self.lim_init_skip,
                                       theta = self.theta)
                    
                    self.assertEqual(len(res), num_chr)
                    
    def test_find_all_chr_diff_identify_scrambled(self):
        
        cell_pts = pd.read_csv('../test/wlc-100hyb-10000kb-4chr-2true-010fp-010fn.csv')
        
        res = find_all_chr(cell_pts, 
                           gene_dist=self.gene_dist, 
                           bin_size = self.bin_size, 
                           nm_per_bp = self.nm_per_bp, 
                           pixel_dist = self.pixel_dist, 
                           l_p_bp = self.l_p_bp,
                           num_skip = self.num_skip,
                           stretch_factor = self.stretch_factor,
                           init_skip_frac = self.init_skip_frac,
                           norm_skip_penalty = self.norm_skip_penalty,
                           lim_init_skip = self.lim_init_skip,
                           theta = self.theta)
        
        self.assertEqual(len(res), 2)
        
#######################################################################################
if __name__ == '__main__':
    unittest.main()