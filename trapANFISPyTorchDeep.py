################################################################################
################################################################################
################################################################################
# Date:             AUG-16-2019
# Institution:      University of Missouri (Columbia, MO)
# Authors:          Blake Ruprecht, Muhammad Islam, and Derek Anderson
#
# DESCRIPTION ------------------------------------------------------------------
#    This is PyTorch code for an Adaptive Neural Fuzzy Inference System (ANFIS)
#
# Notes:
# 1. The below FuzzyNeuron class is a single fuzzy inference system (FIS)
#    What does that mean?
#    - Its a single first-order Takagi Sugeno Kang (TSK) inference system
#    What does that mean?
#    - Each neuron consists of R different IF-THEN rules and the aggregation of
#      their output.
# Coming soon...
#    - We will post cost updates that allow you to do things like
#    - Learn the number of rules R (via an algorithm like DBSCAN or
#      k-means/fcm/pcm with cluster validity)
#
# FOR MORE DETAILS, SEE: -------------------------------------------------------
#
#   Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE
#       Transactions on Systems, Man and Cybernetics, 23 (3), 1993
#
# GNU GENERAL PUBLIC LICENSE ---------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


################################################################################
# LIBRARIES
################################################################################

# OUR LIBRARIES ----------------------------------------------------------------
from dataset_utils import Format_Dataset

# OTHER LIBRARIES --------------------------------------------------------------
import torch
from tqdm import tqdm
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from scipy.linalg import lstsq
import sys
import argparse
from torchvision import datasets


################################################################################
# ANFIS NEURON with TRAP M.F. and multiple different INITs
################################################################################

# ANFIS NEURON CLASS -----------------------------------------------------------
class FuzzyNeuron(torch.nn.Module):

    # Forward Pass -------------------------------------------------------------
    def forward(self,input):

        """ Forward Pass. This function evaluates an ANFIS rule base. If you
        don't already know, when working in PyTorch, keep all functions in
        PyTorch, so autograd will differentiate for you and you don't need to
        override backward pass with custom gradients. """

        batchSize = input.shape[0]
        z = torch.zeros( batchSize, self.R )                                    # our rule outputs
        ants = torch.zeros( self.R, batchSize, A )                              # our rule antecedents
        w = torch.ones( self.R, batchSize )                                     # our rule matching strengths

        sorted,indices = torch.sort(self.abcd)                                  # our sorted trapMF parameters

        for k in range( batchSize ):                                            # do for each sample, "k", in batchSize
            for r in range( self.R ):                                           # do for each rule, "r", in self.R
                z[k,r] = torch.dot(input[k,:],self.rho[r,:-1] ) \
                                                        + self.rho[r,self.A]    # the rule firing
                for n in range( self.A ):                                       # do for each antecedent, "n", in self.A
                    a,b,c,d = sorted[r,n,0],sorted[r,n,1],\
                              sorted[r,n,2],sorted[r,n,3]                       # the params for the trapMF come from the sort
                    ants[r,k,n] = torch.max( torch.tensor([ torch.min( \
                                  torch.tensor([((input[k,n]-a)/(b-a)), 1, \
                                  ((d-input[k,n])/(d-c))]) ), 0]) )             # trapezoidal Membership Function (trapMF)
                w[r,k] = torch.prod( ants[r,k,:] )                              # rule matching strength

        mul = torch.mm(z,w)                                                     # do their mult, but we only want the resultant diag
        diag = torch.diag(mul)                                                  # pull out the diag -> length equal to mini batch size
        wsum = torch.sum(w,dim=0)                                               # now sum across our weights (they are normalizers)

        return diag / (wsum + 0.0000000000001)                                  # now, do that normalization

    # Initialization -----------------------------------------------------------
    def __init__(self, R, A, InitMethod=0, TrainData=[], TrainLabels=[]):

        """ Init Function. This function will initialize the ANFIS parameters:
              (0) randomly
              (1) with k-means clustering
              (2) with k-means clustering and lstmsq rho guess """

        super(FuzzyNeuron,self).__init__()                                      # call parent function

        # InitMethod - Random --------------------------------------------------
        if( InitMethod == 0 ):

            self.R = R                                                          # number of rules
            self.A = A                                                          # number of antecedents
            self.abcd = torch.nn.Parameter(  torch.rand(R,A,4)  )               # trapMF parameters
            self.rho = torch.nn.Parameter( torch.rand(R,A+1) )                  # again, could be any +/-

            return

        # InitMethod - K-Means -------------------------------------------------
        elif( InitMethod == 1 ):

            self.R = R                                                          # number of rules
            self.A = A                                                          # number of antecedents

            NoSamples = TrainData.shape[0]                                      # run the K-Means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', \
                       tol=1e-6, max_iter=500, random_state=0).fit(TrainData)
            print('K-means mean guess')
            print(kmeans.cluster_centers_)

            mu = torch.rand(R,A)                                                # take the centers as our ant's
            for r in range(R):
                for n in range(A):
                    mu[r,n] = kmeans.cluster_centers_[r,n]

            sig = torch.rand(R,A)                                               # now, estimate the variances
            for r in range(R):
                inds = np.where( kmeans.labels_ == r )
                classdata = torch.squeeze( TrainData[inds,:] )
                for n in range(A):
                    sig[r,n] = torch.std( torch.squeeze(classdata[:,n]) )

            abcd = torch.zeros(R,A,4)                                           # build trap function from MU and SIGMA
            for r in range(R):
                for n in range(A):
                    abcd[r,n,0] = mu[r,n] - 5*sig[r,n]                          # c,b are +/- 2 stdDev from MU
                    abcd[r,n,1] = mu[r,n] - 2*sig[r,n]                          # d,a are +/- 5 stdDev from MU
                    abcd[r,n,2] = mu[r,n] + 2*sig[r,n]
                    abcd[r,n,3] = mu[r,n] + 5*sig[r,n]
            self.abcd = torch.nn.Parameter( abcd )

            self.rho = torch.nn.Parameter( torch.rand(R,A+1) )                  # random rhos, could be any +/-

            return

        # InitMethod - K-Means w/ Rho Guess ------------------------------------
        elif( InitMethod == 2 ):

            self.R = R                                                          # number of rules
            self.A = A                                                          # number of antecedents

            NoSamples = TrainData.shape[0]                                      # run the K-Means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', \
                       tol=1e-6, max_iter=500, random_state=0).fit(TrainData)
            print('K-means mean guess')
            print(kmeans.cluster_centers_)

            mu = torch.rand(R,A)                                                # take the centers as our ant's
            for r in range(R):
                for n in range(A):
                    mu[r,n] = kmeans.cluster_centers_[r,n]

            sig = torch.rand(R,A)                                               # now, estimate the variances
            for r in range(R):
                inds = np.where( kmeans.labels_ == r )
                classdata = torch.squeeze( TrainData[inds,:] )
                for n in range(A):
                    sig[r,n] = torch.std( torch.squeeze(classdata[:,n]) )

            abcd = torch.zeros(R,A,4)                                           # build trap function from MU and SIGMA
            for r in range(R):
                for n in range(A):
                    abcd[r,n,0] = mu[r,n] - 5*sig[r,n]                          # c,b are +/- 2 stdDev from MU
                    abcd[r,n,1] = mu[r,n] - 2*sig[r,n]                          # d,a are +/- 5 stdDev from MU
                    abcd[r,n,2] = mu[r,n] + 2*sig[r,n]
                    abcd[r,n,3] = mu[r,n] + 5*sig[r,n]
            self.abcd = torch.nn.Parameter( abcd )

            Ws = torch.zeros(NoSamples,R,A)                                     # now, guess at rhos, using least means squared
            Wprods = torch.ones(NoSamples,R)                                    #  first, calc the constants using mu and sigma
            Wsums = torch.zeros(NoSamples) + 0.00000001
            for s in range(NoSamples):
                for i in range(R):
                    for j in range(A):
                        Ws[s,i,j] = max( min( min( (TrainData[s,j]-abcd[i,j,0])/(abcd[i,j,1]-abcd[i,j,0]), 1 ),\
                                             (abcd[i,j,3]-TrainData[s,j])/(abcd[i,j,3]-abcd[i,j,2]) ) , 0 )
                        Wprods[s,i] = Wprods[s,i] * Ws[s,i,j]                   # each rule is the t-norm (product here) of the MFs
                    Wsums[s] = Wsums[s] + Wprods[s,i]                           # we will normalize by this below

            CoeffMatrix = np.zeros((NoSamples,R*(A+1)))                         # make up our matrix to solve for
            for s in range(NoSamples):
                ctr = 0
                for i in range(R):
                    for j in range(A):
                        CoeffMatrix[s,ctr] = Wprods[s,i] * TrainData[s,j]\
                                             * ( 1.0 / Wsums[s] )
                        ctr = ctr + 1
                    CoeffMatrix[s,ctr] = Wprods[s,i] * 1.0 * (1.0 / Wsums[s])   # now, do for bias term
                    ctr = ctr + 1

            p, res, rnk, s = lstsq(CoeffMatrix, TrainLabels)                    # solve for rho
            rho = torch.zeros(R,A+1)                                            # format and return this as our init
            ctr = 0
            for i in range(R):
                for j in range(A+1):
                    rho[i,j] = p[ctr]
                    ctr = ctr + 1
            print('Rho guess is')
            print(rho)
            self.rho = torch.nn.Parameter( rho )

            return


################################################################################
# DEEP ANFIS LAYERS
################################################################################
class FuzzyNeuronLayer(torch.nn.Module):

    """ Applies multiple ANFIS neurons on a set of inputs
    Args:
        N_in (int) : Number of attributes or sources in the input data
        N_out (int) : Number of outputs produced by ANFIS neurons
        N_rules (int or list): if int, all ANFIS will have the same number of rules
        InitMethods (int or list, optional): set the initialization method. default 0
    Shape:
        Input: (M, N_in)
        Output: (M, N_out),
        where M is the number of samples and N_in is the number of attributes or sources.
    """

    def __init__(self, N_in, N_out, N_rules, InitMethods=0):                    # init function that makes layers

        super(FuzzyNeuronLayer,self).__init__()

        self.N_out = N_out

        if (isinstance(N_rules, int)):
            N_rules = [N_rules for i in range(self.N_out)]

        if (isinstance(InitMethods, int)):
            InitMethods = [InitMethods for i in range(self.N_out)]

        if self.N_out>1:
            self.fuzzyneurons = torch.nn.ModuleList([FuzzyNeuron(N_rules[i], \
                              N_in, InitMethods[i]) for i in range(self.N_out)])
        else:
            self.fuzzyneurons = FuzzyNeuron(N_rules, N_in, InitMethods)

    def forward(self,input):

        if self.N_out>1:
            out = torch.stack([fn(input) for fn in self.fuzzyneurons], dim=1)

        else:
            out = self.fuzzyneurons(input)

        return out

################################################################################
# TRAIN and TEST
################################################################################

if __name__=='__main__':

    # Gaussian MF --------------------------------------------------------------
    def MF(x, mu, sigma):
        return math.exp( - ((x-mu)**2) / (2*sigma**2) )

    # Data ---------------------------------------------------------------------
    NoSamples = 700
    NoPatterns = 7
    x = torch.rand(NoSamples,2)
    print(x.shape[0])
    dl = torch.zeros(NoSamples)

    # Labels -------------------------------------------------------------------
    l = torch.zeros(NoSamples)
    # randomly pick the model parameters
    Membs = torch.rand(NoPatterns,2) # the means
    Stds = torch.rand(NoPatterns,2)*.1 # the standard deviations
    Rows = torch.rand(NoPatterns,3) # these are the linear rules weights
    # sample some random data
    dc = 0
    for r in range(NoPatterns):
        tm = torch.Tensor([Membs[r,0], Membs[r,1]])
        tc = torch.eye(2)
        tc[0,0] = Stds[r,0]*Stds[r,0]
        tc[1,1] = Stds[r,1]*Stds[r,1]
        m = torch.distributions.multivariate_normal.MultivariateNormal(tm,tc)
        for i in range(int(NoSamples/NoPatterns)):
            x[dc,:] = m.sample()
            dl[dc] = r
            dc = dc + 1
    # now, have to fire each rule
    m = torch.zeros(NoSamples)
    ll = torch.rand(NoSamples)
    for i in range(NoSamples):
        mm = 0
        for r in range(NoPatterns):
            if( r == dl[i] ):
               m[r] = 1
            mm = mm + m[r]
            ll[r] = x[i,0] * Rows[r,0] + x[i,1] * Rows[r,1] + Rows[r,2]
        for r in range(NoPatterns):
            l[i] = l[i] + ll[r]*m[r]
        l[i] = l[i] / (mm + 0.0000000000001)
    print('Done with building synthetic data set')

    # Training -----------------------------------------------------------------
    dataset = {'samples': x, 'labels': l}                                       # make up training dataset
    train = Format_Dataset(dataset, choice = 'Train')
    train = torch.utils.data.DataLoader( shuffle = True,
                                             dataset = train,
                                             batch_size = 15 )

    R = 7                                                                       # number of rules to "guess"
    A = 2                                                                       # number of antecedents, needs to be correct!
    initMethod = 2                                                              # init method, 0, 1, 2
    net = FuzzyNeuron(R,A,initMethod,x,l)                                       # create the ANFIS neuron

    criterion = torch.nn.MSELoss()                                              # set up optimization parameters
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.3)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, \
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    NoEpochs = 10
    for epoch in tqdm(range( NoEpochs ),'Training Epoch'):
        for sample, label in train:
            outputs = net(sample)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('Done learning')

    # Visualizaiton ------------------------------------------------------------
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    data = np.asarray(x)

    ABCD = (net.abcd.data).cpu().numpy()
    print(ABCD)

    colormap = ['g', 'k', 'c', 'g', 'k', 'c', 'g', 'k', 'c'

    ]
    for r in range(NoPatterns):
        for d in range( int(NoSamples/NoPatterns) ):
            plt.plot( data[(int(NoSamples/NoPatterns))*r+d,0] , \
                      data[(int(NoSamples/NoPatterns))*r+d,1] , '.', \
                      color=colormap[r] )
    # show rule coverage
    for r in range(R):
        xmiddle = ( ABCD[r,0,2] - ABCD[r,0,1] ) / 2.0 + ABCD[r,0,1]
        ymiddle = ( ABCD[r,1,2] - ABCD[r,1,1] ) / 2.0 + ABCD[r,1,1]
        plt.plot( [ABCD[r,0,0], ABCD[r,0,3]], [ymiddle, ymiddle], 'k--' )
        plt.plot( [xmiddle, xmiddle], [ABCD[r,1,0], ABCD[r,1,3]], 'k--' )
        plt.plot( [ABCD[r,0,1], ABCD[r,0,2]], [ymiddle, ymiddle], 'r--' )
        plt.plot( [xmiddle, xmiddle], [ABCD[r,1,1], ABCD[r,1,2]], 'r-' )
    plt.show()
