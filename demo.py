from __future__ import division
from Numberjack import *
import numpy as np

# load data and shuffle
hardware_data = np.genfromtxt('hardware_data.txt')
np.random.shuffle(hardware_data)

baseball_data = np.genfromtxt('baseball_data.txt')
np.random.shuffle(baseball_data)

religion_data = np.genfromtxt('religion_data.txt')
np.random.shuffle(religion_data)

data = np.concatenate([hardware_data[0:50, :], baseball_data[0:3, :], religion_data[0:3, :]])
num_outliers = 6

# normalize rows
m = np.linalg.norm(data, ord=1, axis=1)
data = np.dot(np.diag(1/m), data)
n = data.shape[0]

# discretization precision (Numberjack only allows integers for variable values)
precision = 100

# pre-compute the pairwise entry-wise inner product
entry_dist = [[[] for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        entry_dist[i][j] = map(lambda x: x.item(), precision*np.abs(data[i]-data[j]))

#print 'Finished precomputing entry-wise distance.'

# select the indices for outliers and normal points
all_index = range(n)
outlier_index = range(n-num_outliers, n) 
normal_index = [i for i in range(n) if i not in outlier_index]

# set bounds for the variables
kmin, kmax = 2, 20
rvalues = (np.arange(0.01, 2.01, 0.01)*precision).astype(int).tolist()
dim = data.shape[1]   

model = Model()

# define variables 
f = VarArray(dim, 'feature_selector_f')      # binary indicator variables used to select features
kN = Variable(kmin, kmax, 'num_neighbors_normal')  
kO = Variable(kmin, kmax, 'num_neighbors_outlier')
rf = Variable(rvalues, 'radius_f')     # values of radius from the predefined list

# auxiliary binary variables indicating within neighborhood
neighborsf = Matrix(n, n, 'neighbor_matrix_f')
for i in range(n):
    for j in range(n):
        model.add(neighborsf[i][j] == (Sum(f, entry_dist[i][j]) <= rf))

# constraints that normal points have at least k neighbors
for i in normal_index:
    model.add(Sum(neighborsf.row[i]) >= kN)

# constraints that outliers have less than k neighbors
for j in outlier_index:
    model.add(Sum(neighborsf.row[j]) < kO)

# constrain the number of features selected
model.add(Sum(f) <= 10)

# maximize the difference between number of neighbors in normal and outliers
k_diff = Variable(0, kmax, 'k_diff')
model.add(k_diff == (kN - kO))
model.add(k_diff >= 0)
model.add(Maximize(k_diff))

# solve
solver = model.load('Gurobi')
solver.setVerbosity(1)      # print optimization steps
#solver.setTimeLimit(3*3600)   # set amount of time allowed
#solver.setThreadCount(12)

if solver.solve():
    print '-'*50
    print 'A solution was found. Variable values are as follows.'
    print '-'*50

    print 'Feature seletor:'
    print f
    print 'Number of neighbors for normal points and outliers:'
    print kN, kO
    print 'Radius of neighborhood:'
    print rf
    #print neighborsf


