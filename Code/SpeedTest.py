#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:10:02 2020

@author: BenjaminSuter
"""

from MCSpeedTest import MonteCarloENTROPY

data1 = "./Samples/4probes_auto_samples_Gaussian.npy"
data2 = "./Samples/4probes_cross_samples_Gaussian.npy"

#steps = [5, 10, 100, 500, 1000, 5000, 10000, 50000, 100000]
steps = [10000]

if __name__ == '__main__':
    time = {}
    for step in steps:
        time[step] = str(MonteCarloENTROPY(data1, data2, step))
    
    print(time)