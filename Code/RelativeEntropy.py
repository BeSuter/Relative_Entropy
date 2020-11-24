#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  23 11:44:14 2020

@author: BenjaminSuter
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from IterCompRelEnt import IterativeEntropy, LoadAndComputeEntropy
from IterCompRelEnt import RemoveScatter
from IterCompRelEnt import MonteCarloENTROPY, LogarithmicScore, EntropyConsistencyTest
from Plotter import GaussianityTest, Chi2ComparisonPlot, ConvergencePlot
from Plotter import RelativeEntropyBarPlot, GetDistContourPlot
from Plotter import MonteCarloHistogram
from surprise import Surprise
import yaml

#Read Data
#
#
# Two probes auto-correlated data
tt_kk_sim = "../Samples/tt_kk_samples_sim.npy"
#350000
'''GetDistContourPlot(tt_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_kk_sim.pdf', burning=350000)'''
tt_kk_gauss = "../Samples/tt_kk_samples_Gaussian.npy"
#350000
'''GetDistContourPlot(tt_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_kk_gauss.pdf', burning=350000)'''
tt_gg_sim = "../Samples/tt_gg_samples_sim.npy"
#350000
'''GetDistContourPlot(tt_gg_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_gg_sim.pdf', burning=350000)'''
tt_gg_gauss = "../Samples/tt_gg_samples_Gaussian.npy"
#350000
'''GetDistContourPlot(tt_gg_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_gg_gauss.pdf', burning=350000)'''
tt_dd_sim = "../Samples/tt_dd_samples_sim.npy"
#350000
'''GetDistContourPlot(tt_dd_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_dd_sim.pdf', burning=350000)'''
tt_dd_gauss = "../Samples/tt_dd_samples_Gaussian.npy"
#300000
'''GetDistContourPlot(tt_dd_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_dd_gauss.pdf', burning=300000)'''
#gg_kk_sim = "./Samples/gg_kk_samples_sim.npy"
#gg_kk_gauss = "./Samples/gg_kk_samples_Gaussian.npy"
dd_kk_sim = "../Samples/dd_kk_samples_sim.npy"
#320000
'''GetDistContourPlot(dd_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_kk_sim.pdf', burning=320000)'''
dd_kk_gauss = "../Samples/dd_kk_samples_Gaussian.npy"
#250000
'''GetDistContourPlot(dd_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_kk_gauss.pdf', burning=250000)'''
dd_gg_sim = "../Samples/dd_gg_samples_sim.npy"
#300000
'''GetDistContourPlot(dd_gg_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_gg_sim.pdf', burning=300000)'''
dd_gg_gauss = "../Samples/dd_gg_samples_Gaussian.npy"
#300000
'''tmp = np.load(dd_gg_gauss)
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
GetDistContourPlot(dd_gg_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_gg_gauss.pdf', burning=pri_burn)'''
#
# Two probes cross_correlated data
tt_tk_kk_sim = "../Samples/tt_tk_kk_samples_sim.npy"
#210000
'''GetDistContourPlot(tt_tk_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_tk_kk_sim.pdf', burning=210000)'''
tt_tk_kk_gauss = "../Samples/tt_tk_kk_samples_Gaussian.npy"
#220000
'''GetDistContourPlot(tt_tk_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_tk_kk_gauss.pdf', burning=220000)'''
tt_tg_gg_sim = "../Samples/tt_tg_gg_samples_sim.npy"
#280000
'''GetDistContourPlot(tt_tg_gg_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_sim.pdf', burning=280000)'''
tt_tg_gg_gauss = "../Samples/tt_tg_gg_samples_Gaussian.npy"
#220000
'''GetDistContourPlot(tt_tg_gg_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_gauss.pdf', burning=220000)'''
tt_td_dd_sim = "../Samples/tt_td_dd_samples_sim.npy"
#280000
'''GetDistContourPlot(tt_td_dd_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_sim.pdf', burning=280000)'''
tt_td_dd_gauss = "../Samples/tt_td_dd_samples_Gaussian.npy"
#220000
'''GetDistContourPlot(tt_td_dd_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_gauss.pdf', burning=220000)'''
dd_dk_kk_sim = "../Samples/dd_dk_kk_samples_sim.npy"
#150000
'''GetDistContourPlot(dd_dk_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_dk_kk_sim.pdf', burning=150000)'''
dd_dk_kk_gauss = "../Samples/dd_dk_kk_samples_Gaussian.npy"
#180000
'''GetDistContourPlot(dd_dk_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_dk_kk_gauss.pdf', burning=180000)'''
dd_dg_gg_sim = "../Samples/dd_dg_gg_samples_sim.npy"
#160000
'''GetDistContourPlot(dd_dg_gg_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_dg_gg_sim.pdf', burning=160000)'''
dd_dg_gg_gauss = "../Samples/dd_dg_gg_samples_Gaussian.npy"
#120000
'''GetDistContourPlot(dd_dg_gg_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2"],
                   'test plot',
                   '../Plots/Contour/dd_dg_gg_gauss.pdf', burning=120000)'''
#
# Three probes auto-correlated
tt_gg_kk_sim = "../Samples/tt_gg_kk_samples_sim.npy"
#400000
'''GetDistContourPlot(tt_gg_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_gg_kk_sim.pdf', burning=400000)'''
tt_gg_kk_gauss = "../Samples/tt_gg_kk_samples_Gaussian.npy"
'''GetDistContourPlot(tt_gg_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_gg_kk_gauss.pdf', burning=450000)'''
#450000
tt_dd_kk_sim = "../Samples/tt_dd_kk_samples_sim.npy"
#400000
'''GetDistContourPlot(tt_dd_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_dd_kk_sim.pdf', burning=400000)'''
tt_dd_kk_gauss = "../Samples/tt_dd_kk_samples_Gaussian.npy"
#450000
'''GetDistContourPlot(tt_dd_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_dd_kk_gauss.pdf', burning=450000)'''
tt_dd_gg_sim = "../Samples/tt_dd_gg_samples_sim.npy"
#400000
'''GetDistContourPlot(tt_dd_gg_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_dd_gg_sim.pdf', burning=400000)'''
tt_dd_gg_gauss = "../Samples/tt_dd_gg_samples_Gaussian.npy"

#450000
'''GetDistContourPlot(tt_dd_gg_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_dd_gg_gauss.pdf', burning=450000)'''
dd_gg_kk_sim = "../Samples/dd_gg_kk_samples_sim.npy"
#400000
'''GetDistContourPlot(dd_gg_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/dd_gg_kk_sim.pdf', burning=400000)'''
dd_gg_kk_gauss = "../Samples/dd_gg_kk_samples_Gaussian.npy"
#400000
'''GetDistContourPlot(dd_gg_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/dd_gg_kk_gauss.pdf', burning=400000)'''
#
# Three probes cross-correlated
tt_tg_gg_gk_kk_sim = "../Samples/tt_tg_gg_gk_kk_samples_sim.npy"
#350000
'''GetDistContourPlot(tt_tg_gg_gk_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_gk_kk_sim.pdf', burning=350000)'''
tt_tg_gg_gk_kk_gauss = "../Samples/tt_tg_gg_gk_kk_samples_Gaussian.npy"
#350000
'''GetDistContourPlot(tt_tg_gg_gk_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_gk_kk_gauss.pdf', burning=350000)'''
tt_td_dd_dk_kk_sim = "../Samples/tt_td_dd_dk_kk_samples_sim.npy"
#250000
'''GetDistContourPlot(tt_td_dd_dk_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dk_kk_sim.pdf', burning=250000)'''
tt_td_dd_dk_kk_gauss = "../Samples/tt_td_dd_dk_kk_samples_Gaussian.npy"
#250000
'''GetDistContourPlot(tt_td_dd_dk_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dk_kk_gauss.pdf', burning=250000)'''
tt_td_dd_dg_gg_sim = "../Samples/tt_td_dd_dg_gg_samples_sim.npy"
#250000
'''GetDistContourPlot(tt_td_dd_dg_gg_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dg_gg_sim.pdf', burning=250000)'''
tt_td_dd_dg_gg_gauss = "../Samples/tt_td_dd_dg_gg_samples_Gaussian.npy"
#250000
'''GetDistContourPlot(tt_td_dd_dg_gg_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dg_gg_gauss.pdf', burning=250000)'''
dd_dg_gg_gk_kk_sim = "../Samples/dd_dg_gg_gk_kk_samples_sim.npy"
#400000
'''GetDistContourPlot(dd_dg_gg_gk_kk_sim,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/dd_dg_gg_gk_kk_sim.pdf', burning=400000)'''
dd_dg_gg_gk_kk_gauss = "../Samples/dd_dg_gg_gk_kk_samples_Gaussian.npy"
#400000
'''GetDistContourPlot(dd_dg_gg_gk_kk_gauss,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8", "m_1", "m_2", "m_3"],
                   'test plot',
                   '../Plots/Contour/dd_dg_gg_gk_kk_gauss.pdf', burning=400000)'''
#
# 4 probes
#SamplePath = "./Samples/tt_dd_gg_kk_samples.npy"
Auto_Gaussian = "../Samples/4probes_auto_samples_Gaussian.npy"
Auto_Sim = "../Samples/4probes_auto_samples_sim.npy"
Cross_Gaussian = "../Samples/4probes_cross_samples_Gaussian.npy"
Cross_Sim = "../Samples/4probes_cross_samples_sim.npy"
#
#------------------------------------------------------------------------------
#Probes with no nuisance parameters
#
#Two probes auto-correlated
tt_kk_sim_noN = "../Samples/tt_kk_samples_sim_nonuisance.npy"

#480000
'''GetDistContourPlot(tt_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_kk_sim_noN.pdf', burning=480000)'''
tt_kk_gauss_noN = "../Samples/tt_kk_samples_Gaussian_nonuisance.npy"
#520000
'''GetDistContourPlot(tt_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_kk_gauss_noN.pdf', burning=520000)'''
tt_gg_sim_noN = "../Samples/tt_gg_samples_sim_nonuisance.npy"
#500000
'''GetDistContourPlot(tt_gg_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_gg_sim_noN.pdf', burning=500000)'''
tt_gg_gauss_noN = "../Samples/tt_gg_samples_Gaussian_nonuisance.npy"
#510000
'''GetDistContourPlot(tt_gg_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_gg_gauss_noN.pdf', burning=510000)'''
tt_dd_sim_noN = "../Samples/tt_dd_samples_sim_nonuisance.npy"
#530000
'''GetDistContourPlot(tt_dd_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_dd_sim_noN.pdf', burning=530000)'''
tt_dd_gauss_noN = "../Samples/tt_dd_samples_Gaussian_nonuisance.npy"
#530000
'''GetDistContourPlot(tt_dd_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_dd_gauss_noN.pdf', burning=530000)'''
gg_kk_sim_noN = "../Samples/gg_kk_samples_sim_nonuisance.npy"
#380000
'''GetDistContourPlot(gg_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/gg_kk_sim_noN.pdf', burning=380000)'''
gg_kk_gauss_noN = "../Samples/gg_kk_samples_Gaussian_nonuisance.npy"
#400000
'''GetDistContourPlot(gg_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/gg_kk_gauss_noN.pdf', burning=400000)'''
dd_kk_sim_noN = "../Samples/dd_kk_samples_sim_nonuisance.npy"
#400000
'''GetDistContourPlot(dd_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/dd_kk_sim_noN.pdf', burning=400000)'''
dd_kk_gauss_noN = "../Samples/dd_kk_samples_Gaussian_nonuisance.npy"
#400000
'''GetDistContourPlot(dd_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/dd_kk_gauss_noN.pdf', burning=400000)'''

#Two probes cross-correlated
tt_tk_kk_sim_noN = "../Samples/tt_tk_kk_samples_sim_nonuisance.npy"
#320000
'''GetDistContourPlot(tt_tk_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_tk_kk_sim_noN.pdf', burning=320000)'''
tt_tk_kk_gauss_noN = "../Samples/tt_tk_kk_samples_Gaussian_nonuisance.npy"
#320000
'''GetDistContourPlot(tt_tk_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_tk_kk_gauss_noN.pdf', burning=320000)'''
tt_tg_gg_sim_noN = "../Samples/tt_tg_gg_samples_sim_nonuisance.npy"
#430000
'''GetDistContourPlot(tt_tg_gg_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_sim_noN.pdf', burning=430000)'''
tt_tg_gg_gauss_noN = "../Samples/tt_tg_gg_samples_Gaussian_nonuisance.npy"
#310000
'''GetDistContourPlot(tt_tg_gg_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_gauss_noN.pdf', burning=310000)'''
tt_td_dd_sim_noN = "../Samples/tt_td_dd_samples_sim_nonuisance.npy"
#350000
'''GetDistContourPlot(tt_td_dd_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_sim_noN.pdf', burning=350000)'''
tt_td_dd_gauss_noN = "../Samples/tt_td_dd_samples_Gaussian_nonuisance.npy"
#360000
'''GetDistContourPlot(tt_td_dd_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_gauss_noN.pdf', burning=360000)'''
dd_dk_kk_sim_noN = "../Samples/dd_dk_kk_samples_sim_nonuisance.npy"
#210000
'''GetDistContourPlot(dd_dk_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/dd_dk_kk_sim_noN.pdf', burning=210000)'''
dd_dk_kk_gauss_noN = "../Samples/dd_dk_kk_samples_Gaussian_nonuisance.npy"
#210000
'''GetDistContourPlot(dd_dk_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/dd_dk_kk_gauss_noN.pdf', burning=210000)'''

#Three probes auto-correlated
tt_gg_kk_sim_noN = "../Samples/tt_gg_kk_samples_sim_nonuisance.npy"
#380000
'''GetDistContourPlot(tt_gg_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_gg_kk_sim_noN.pdf', burning=380000)'''
tt_gg_kk_gauss_noN = "../Samples/tt_gg_kk_samples_Gaussian_nonuisance.npy"
#360000
'''GetDistContourPlot(tt_gg_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_gg_kk_gauss_noN.pdf', burning=360000)'''

tt_dd_kk_sim_noN = "../Samples/tt_dd_kk_samples_sim_nonuisance.npy"
#360000
'''GetDistContourPlot(tt_dd_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_dd_kk_sim_noN.pdf', burning=360000)'''
tt_dd_kk_gauss_noN = "../Samples/tt_dd_kk_samples_Gaussian_nonuisance.npy"
#360000
'''GetDistContourPlot(tt_dd_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_dd_kk_gauss_noN.pdf', burning=360000)'''
tt_dd_gg_sim_noN = "../Samples/tt_dd_gg_samples_sim_nonuisance.npy"
#360000
'''GetDistContourPlot(tt_dd_gg_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_dd_gg_sim_noN.pdf', burning=360000)'''
tt_dd_gg_gauss_noN = "../Samples/tt_dd_gg_samples_Gaussian_nonuisance.npy"
#360000
'''GetDistContourPlot(tt_dd_gg_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_dd_gg_gauss_noN.pdf', burning=360000)'''

#Three probes cross-correlated
tt_tg_gg_gk_kk_sim_noN = "../Samples/tt_tg_gg_gk_kk_samples_sim_nonuisance.npy"
#230000
'''GetDistContourPlot(tt_tg_gg_gk_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_gk_kk_sim_noN.pdf', burning=230000)'''
tt_tg_gg_gk_kk_gauss_noN = "../Samples/tt_tg_gg_gk_kk_samples_Gaussian_nonuisance.npy"
#140000
'''GetDistContourPlot(tt_tg_gg_gk_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_tg_gg_gk_kk_gauss_noN.pdf', burning=140000)'''
tt_td_dd_dk_kk_sim_noN = "../Samples/tt_td_dd_dk_kk_samples_sim_nonuisance.npy"
#220000
'''GetDistContourPlot(tt_td_dd_dk_kk_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dk_kk_sim_noN.pdf', burning=220000)'''
tt_td_dd_dk_kk_gauss_noN = "../Samples/tt_td_dd_dk_kk_samples_Gaussian_nonuisance.npy"
#230000
'''GetDistContourPlot(tt_td_dd_dk_kk_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dk_kk_gauss_noN.pdf', burning=230000)'''
tt_td_dd_dg_gg_sim_noN = "../Samples/tt_td_dd_dg_gg_samples_sim_nonuisance.npy"
#220000
'''GetDistContourPlot(tt_td_dd_dg_gg_sim_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dg_gg_sim_noN.pdf', burning=220000)'''
tt_td_dd_dg_gg_gauss_noN = "../Samples/tt_td_dd_dg_gg_samples_Gaussian_nonuisance.npy"
#160000
'''GetDistContourPlot(tt_td_dd_dg_gg_gauss_noN,
                   ["h", "\Omega_m", "\Omega_b", "n_s",
                    "\sigma_8"],
                   'test plot',
                   '../Plots/Contour/tt_td_dd_dg_gg_gauss_noN.pdf', burning=160000)'''

#Additional noNuisance samples
#Two Probes
dd_gg_sim_noN = '../Samples/dd_gg_samples_sim_nonuisance.npy'
dd_dg_gg_sim_noN = '../Samples/dd_dg_gg_samples_sim_nonuisance.npy'

dd_gg_gauss_noN = '../Samples/dd_gg_samples_Gaussian_nonuisance.npy'
dd_dg_gg_gauss_noN = '../Samples/dd_dg_gg_samples_Gaussian_nonuisance.npy'

#Three Probes
dd_gg_kk_sim_noN = '../Samples/dd_gg_kk_samples_sim_nonuisance.npy'
dd_dg_gg_gk_kk_sim_noN = '../Samples/dd_dg_gg_gk_kk_samples_sim_nonuisance.npy'

dd_gg_kk_gauss_noN = '../Samples/dd_gg_kk_samples_Gaussian_nonuisance.npy'
dd_dg_gg_gk_kk_gauss_noN = '../Samples/dd_dg_gg_gk_kk_samples_Gaussian_nonuisance.npy'

#Four Probes
Auto_Sim_noN = '../Samples/tt_dd_gg_kk_samples_sim_nonuisance.npy'
Cross_Sim_noN = '../Samples/tt_td_dd_dg_gg_gk_kk_samples_sim_nonuisance.npy'

Auto_Gauss_noN = '../Samples/tt_dd_gg_kk_samples_Gaussian_nonuisance.npy'
Cross_sim_noN = '../Samples/tt_td_dd_dg_gg_gk_kk_samples_Gaussian_nonuisance.npy'


#
#
#Collected data
TwoProbesData = [(tt_kk_sim, tt_tk_kk_sim), (tt_kk_gauss, tt_tk_kk_gauss),
                 (tt_gg_sim, tt_tg_gg_sim), (tt_gg_gauss, tt_tg_gg_gauss),
                 (tt_dd_sim, tt_td_dd_sim), (tt_dd_gauss, tt_td_dd_gauss),
                 (dd_kk_sim, dd_dk_kk_sim), (dd_kk_gauss, dd_dk_kk_gauss),
                 (dd_gg_sim, dd_dg_gg_sim), (dd_gg_gauss, dd_dg_gg_gauss),
                 (tt_kk_sim_noN, tt_tk_kk_sim_noN),
                 (tt_kk_gauss_noN, tt_tk_kk_gauss_noN),
                 (tt_gg_sim_noN, tt_tg_gg_sim_noN),
                 (tt_gg_gauss_noN, tt_tg_gg_gauss_noN),
                 (tt_dd_sim_noN, tt_td_dd_sim_noN),
                 (tt_dd_gauss_noN, tt_td_dd_gauss_noN),
                 (dd_kk_sim_noN, dd_dk_kk_sim_noN),
                 (dd_kk_gauss_noN, dd_dk_kk_gauss_noN),
                 (dd_gg_sim_noN, dd_dg_gg_sim_noN),
                 (dd_gg_gauss_noN, dd_dg_gg_gauss_noN)]

ThreeProbesData = [(tt_dd_gg_gauss_noN,tt_td_dd_dg_gg_gauss_noN),
                   (tt_dd_gg_sim_noN,tt_td_dd_dg_gg_sim_noN),
                   (tt_dd_kk_gauss_noN,tt_td_dd_dk_kk_gauss_noN),
                   (tt_dd_kk_sim_noN,tt_td_dd_dk_kk_sim_noN),
                   (tt_gg_kk_gauss_noN,tt_tg_gg_gk_kk_gauss_noN),
                   (tt_gg_kk_sim_noN,tt_tg_gg_gk_kk_sim_noN),
                   (dd_gg_kk_sim_noN, dd_dg_gg_gk_kk_sim_noN),
                   (dd_gg_kk_gauss_noN, dd_dg_gg_gk_kk_gauss_noN),
                   (tt_gg_kk_sim,tt_tg_gg_gk_kk_sim),
                   (tt_dd_kk_gauss,tt_td_dd_dk_kk_gauss),
                   (tt_gg_kk_gauss,tt_tg_gg_gk_kk_gauss),
                   (tt_dd_gg_sim,tt_td_dd_dg_gg_sim),
                   (tt_dd_gg_gauss,tt_td_dd_dg_gg_gauss),
                   (dd_gg_kk_sim,dd_dg_gg_gk_kk_sim),
                   (dd_gg_kk_gauss,dd_dg_gg_gk_kk_gauss),
                   (tt_dd_kk_sim,tt_td_dd_dk_kk_sim)]

FourProbesData = [(Auto_Gaussian, Cross_Gaussian),
                  (Auto_Sim, Cross_Sim),
                  (Auto_Sim_noN, Cross_Sim_noN),
                  (Auto_Gauss_noN, Cross_sim_noN)]



def EntropyComparison(data, avrg, target, burn_in=None, MCsteps=100000):
    """
    Will compute the relative entropy via monte carlo integration and the
    Surprise package and write the results to a YAML file.
    
    Args:
        data (list): A list of tuples containing the path to the prior and 
                     posterior distributions
        avrg (int): Number of computations over which to average the results
                    of the MC entropy and the gauss approximated entropy
        target (str): Target to where to save the YAML file
        MCsteps (int): Number of sample points used when computing the entropy
                       via Monte Carlo integration
    """
    EntropyValues = {'Names': ['MC Entropy', 'MC Error', 'Gauss Entropy',
                               'Gauss Error', 'Expected Entropy','Surprise',
                               'sD', 'Gauss Entropy converged',
                               'Gauss Error converged',
                               'Expected Entropy converged','Surprise converged',
                               'sD converged', 'Entropy no BC',
                               'Expected Entropy no BC']}
    for names in data:
        tmp_mc = []
        tmp_gauss_ent = []
        tmp_gauss_exp = []
        tmp_gauss_S = []
        tmp_gauss_sD = []
        tmp_gauss_conv_ent = []
        tmp_gauss_conv_exp = []
        tmp_gauss_conv_S = []
        tmp_gauss_conv_sD = []
        
        size1 = avrg
        size2 = avrg
        for i in range(avrg):
            print(f"Computing MonteCarloENTROPY {i+1}/{avrg}")
            tmp_mc.append(MonteCarloENTROPY(names[0],names[1], MCsteps,
                                            pri_burn=burn_in, post_burn=burn_in))
            print(f"Computing gaussian approximation with no convergence {i+1}/{avrg}")
            gauss_approx = LoadAndComputeEntropy(names[0], names[1], steps=1,
                                                 pri_burn=burn_in, post_burn=burn_in)
            tmp = [gauss_approx[2][-1], gauss_approx[3][-1],
                   gauss_approx[-3][-1], gauss_approx[-4][-1]]
            if not gauss_approx[-1] and not None in tmp:
                tmp_gauss_ent.append(gauss_approx[2][-1])
                tmp_gauss_exp.append(gauss_approx[3][-1])
                tmp_gauss_S.append(gauss_approx[-3][-1])
                tmp_gauss_sD.append(gauss_approx[-2][-1])
            else:
                size1 -= 1
            print(f"Computing gaussian approximation with convergence {i+1}/{avrg}")
            gauss_approx_conv = LoadAndComputeEntropy(names[0], names[1], steps=300)
            tmp = [gauss_approx_conv[2][-1], gauss_approx_conv[3][-1], 
                   gauss_approx_conv[-3][-1], gauss_approx_conv[-4][-1]]
            if not gauss_approx_conv[-1] and not None in tmp:
                tmp_gauss_conv_ent.append(gauss_approx_conv[2][-1])
                tmp_gauss_conv_exp.append(gauss_approx_conv[3][-1])
                tmp_gauss_conv_S.append(gauss_approx_conv[-3][-1])
                tmp_gauss_conv_sD.append(gauss_approx_conv[-2][-1])
            else:
                size2 -= 1
            
        mc_error = np.around(np.sqrt(np.var(tmp_mc)), decimals=3)
        try:
            gauss_error = np.around(np.sqrt(np.var(tmp_gauss_ent)), decimals=3)
        except:
            gauss_error = None
        try:
            gauss_conv_error = np.around(np.sqrt(np.var(tmp_gauss_conv_ent)), decimals=3)
        except:
            gauss_conv_error = None
        
        mc_ent = np.around(sum(tmp_mc)/avrg, decimals=3)
        if size1 > 0:
            gauss_ent = np.around(sum(tmp_gauss_ent)/size1, decimals=3)
            gauss_exp = np.around(sum(tmp_gauss_exp)/size1, decimals=3)
            gauss_S = np.around(sum(tmp_gauss_S)/size1, decimals=3)
            gauss_sD = np.around(sum(tmp_gauss_sD)/size1, decimals=3)
        else:
            gauss_ent = None
            gauss_exp = None
            gauss_S = None
            gauss_sD = None
        if size2 > 0:
            gauss_conv_ent = np.around(sum(tmp_gauss_conv_ent)/size2, decimals=3)
            gauss_conv_exp = np.around(sum(tmp_gauss_conv_exp)/size2, decimals=3)
            gauss_conv_S = np.around(sum(tmp_gauss_conv_S)/size2, decimals=3)
            gauss_conv_sD = np.around(sum(tmp_gauss_conv_sD)/size2, decimals=3)
        else:
            gauss_conv_ent = None
            gauss_conv_exp = None
            gauss_conv_S = None
            gauss_conv_sD = None
        
        if burn_in == None:
            burn = 0.5
        else:
            burn = burn_in
        tmp = np.load(names[0])
        pri_burn = int(np.shape(tmp)[0] * burn)
        del(tmp)
        data1 = np.load(names[0])[pri_burn:,:5]
        
        tmp = np.load(names[1])
        pri_burn = int(np.shape(tmp)[0] * burn)
        del(tmp)
        data2 = np.load(names[1])[pri_burn:,:5]
        sc = Surprise()
        try:
            vals = sc(data1, data2, mode='add')[0:2]
            vals = np.around(vals, decimals=3)
        except AssertionError:
            vals = [None, None]
        ent_noBC = vals[0]
        EXPent_noBC = vals[1]
        del(data1)
        del(data2)
        
        name = names[0][10:].replace('.npy', '')
        EntropyValues[name] = ['%s' % mc_ent, 
                               '%s' % mc_error,
                               '%s' % gauss_ent,
                               '%s' % gauss_error,
                               '%s' % gauss_exp,
                               '%s' % gauss_S,
                               '%s' % gauss_sD, 
                               '%s' % gauss_conv_ent,
                               '%s' % gauss_conv_error,
                               '%s' % gauss_conv_exp,
                               '%s' % gauss_conv_S,
                               '%s' % gauss_conv_sD,
                               '%s' % ent_noBC,
                               '%s' % EXPent_noBC]
    
    with open(target, 'w') as yaml_file:
        yaml.dump(EntropyValues, stream=yaml_file, default_flow_style=False)



gaussianity_hist = []
#TDGKs = LoadAndComputeEntropy(FourProbesData[1][0], FourProbesData[1][1], steps=300)
#TGKg = LoadAndComputeEntropy(ThreeProbesData[-5][0], ThreeProbesData[-5][1], steps=300)

'''tmp = np.load(FourProbesData[1][1])
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
rawCrossTDGKs = np.load(FourProbesData[1][1])[pri_burn:,:5]

tmp = np.load(FourProbesData[1][0])
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
rawAutoTDGKs = np.load(FourProbesData[1][0])[pri_burn:,:5]'''

'''tmp = np.load(ThreeProbesData[-5][1])
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
rawCrossTGKg = np.load(ThreeProbesData[-5][1])[pri_burn:,:5]

tmp = np.load(ThreeProbesData[-5][0])
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
rawAutoTGKg = np.load(ThreeProbesData[-5][0])[pri_burn:,:5]'''

'''tmp = np.load(ThreeProbesData[-2][1])
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
rawCrossDGKs = np.load(ThreeProbesData[-2][1])[pri_burn:,:5]

tmp = np.load(ThreeProbesData[-2][0])
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
rawAutoDGKs = np.load(ThreeProbesData[-2][0])[pri_burn:,:5]'''


#gaussianity_hist.append(GaussianityTest(TGKg[1].T))
#gaussianity_hist.append(GaussianityTest(TGKg[0].T))
#gaussianity_hist.append(GaussianityTest(rawCrossDGKs))
#gaussianity_hist.append(GaussianityTest(rawAutoDGKs))

'''labels = ['Auto+Cross TGKg Box-Cox', 'Auto TGKg Box-Cox',
          'Auto+Cross TGKg', 'Auto TGKg']'''
labels = ['Auto+Cross DGKs', 'Auto DGKs']

'''Chi2ComparisonPlot(gaussianity_hist,
                   './Plots/GaussianityTest/DGKs.pdf',
                   labels=labels)'''

'''gaussianity_hist = []
labels = []

for data in TwoProbesData[9:]:
    gauss_approx = LoadAndComputeEntropy(data[0], data[1], steps=2)
    
    gaussianity_hist.append(GaussianityTest(gauss_approx[1].T))
    labels.append(data[0][10:].replace('.npy', ''))
Chi2ComparisonPlot(gaussianity_hist,
                   './Plots/GaussianityTest/Half_Posterior2.pdf',
                   labels=labels)

gaussianity_hist = []
labels = []

for data in TwoProbesData[9:]:
    gauss_approx = LoadAndComputeEntropy(data[0], data[1], steps=2)
    
    gaussianity_hist.append(GaussianityTest(gauss_approx[0].T))
    labels.append(data[0][10:].replace('.npy', ''))
Chi2ComparisonPlot(gaussianity_hist,
                   './Plots/GaussianityTest/Half_Prior2.pdf',
                   labels=labels)'''


"""if __name__ == "__main__":
    args = sys.argv[1:]
    data_type = args[0]
    avrg = int(args[1])
    MCsteps = int(args[2])
    try:
        burn_in = float(args[3])
    except:
        burn_in = None
    if data_type == "TwoProbesData":
        data = TwoProbesData
    elif data_type == "ThreeProbesData":
        data = ThreeProbesData
    elif data_type == "FourProbesData":
        data = FourProbesData
    print(f"\nExecuting Entropy Comparison for {data_type} with burn-in={burn_in} avrg={avrg} and MCstep={MCsteps}\n")
    EntropyComparison(data, avrg, f"../YAML/{data_type}_burn={burn_in}_avrg={avrg}_MCsteps={MCsteps}TEST.yml",
                      burn_in=burn_in, MCsteps=MCsteps)"""

if __name__ == "__main__":
    mc = LoadAndComputeEntropy(Auto_Sim, Cross_Sim, steps=3)
    print(mc[:][0])