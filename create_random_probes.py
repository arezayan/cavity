
"""
Created on Wed Aug  7 10:37:05 2024

@author: zippi
"""

#how to creat random points

import random
import numpy as np
import pandas as pd

def rand_points(minx,maxx,miny,maxy,ndata,dim):
    point = np.zeros((ndata,dim+1))
    for i in range(ndata):
        a = random.uniform(minx,maxx)
        b = random.uniform(miny,maxy)
        point[i,0] = a
        point[i,1] = b
        point[i,2] = 0.05
    return pd.DataFrame(point).to_excel("output.xlsx",index_label=None)


rand_points(0, 1, 0, 0.1, 130, 2)