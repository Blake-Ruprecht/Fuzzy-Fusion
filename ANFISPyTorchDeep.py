##############################################################################
##############################################################################
##############################################################################
# University of Missouri-Columbia
#
# 7/5/2019
#
# Author: Blake Ruprecht and Muhammad Islam and Derek Anderson
#
# Description:
#  This is PyTorch code for an adaptive neural fuzzy inference system (ANFIS)
#  Notes:
#   1. The below FuzzyNeuron class is a single fuzzy inference system (FIS)
#    What does that mean? 
#      Its a single first order (aka linear) Takagi Sugeno Kang (TSK) inference system
#    What does that mean?
#      Each neuron is R different IF-THEN rules and the aggregation of their output
#   2. At the bottom of this file, we also show how to extend what we did for "deep ANFIS"
#    What does that mean?
#      It means you can make "layers" of ANFIS neurons (rule bases)
#  Coming soon...
#   We will post cost updates that allow you to do things like
#    Learn the number of rules R (via an algorithm like DBSCAN or k-means/fcm/pcm with cluster validity)
#
##############################################################################
#
# For more details, see:
# Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE Transactions on Systems, Man and Cybernetics, 23 (3), 1993
#
##############################################################################
#
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
#
##############################################################################
##############################################################################
##############################################################################

# Others Libraries 
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
# Our Libs
from dataset_utils import Format_Dataset

# FUZZY INFERENCE SYSTEM NEURON ------------------------------------------------
class FuzzyNeuron(torch.nn.Module):

    ############################################
    ############################################
    ############################################
    ############################################
    def forward(self,input):
        
        """ forward pass """

        # this function evaluates an ANFIS (aka, a rule base)
        # if you do not already know, in PyTorch, keep it all in PyTorch, autograd will auto differentiate for you 
        #   meaning, you do not have to overide the backward and put in your own custom gradients
        
        z = torch.zeros( input.shape[0], self.R )    # our rule outputs
        w = torch.ones( self.R, input.shape[0] )     # our rule matching strengths 

        for k in range( input.shape[0] ):            # go over all samples (make linear algebra in future so faster; did for readability)
            for r in range( self.R ):                # do for each rule (make linear algebra in future; did for readability)
                z[k,r] = torch.dot( input[k,:], self.rho[r,:-1] ) + self.rho[r,self.A]  # the rule firing
                w[r,k] = torch.prod( torch.exp( (-((input[k,:]-self.mu[r,:])**2)) / ((2*self.sigma[r,:]**2)) ) )  # membership fx calc
        mul = torch.mm(z,w)                          # do their mult, but we only want the resultant diag
        diag = torch.diag(mul)                       # pull out the diag -> length equal to mini batch size
        wsum = torch.sum(w,dim=0)                    # now sum across our weights (they are normalizers)
        
        return diag / (wsum + 0.0000000000001)       # now, do that normalization

    ############################################
    ############################################
    ############################################
    ############################################
    def __init__(self, R, A, InitMethod=0, TrainData=[], TrainLabels=[]): 
        
        """ init function """

        super(FuzzyNeuron,self).__init__()                         # call parent function
        
        if( InitMethod == 0 ): # random init
        
            self.R = R                                                 # our number of rules
            self.A = A                                                 # our number of antecedents
            self.mu = torch.nn.Parameter( torch.rand(R,A) )            # in [0,1], but could be +/-, whatever
            self.sigma = torch.nn.Parameter( torch.rand(R,A)*0.2 )     # in [0,1], but don't let it get too small semantically
            self.rho = torch.nn.Parameter( torch.rand(R,A+1) )         # again, could be any +/-
        
            return
            
        elif( InitMethod == 1 ): # k-means based init
        
            self.R = R                                  # our number of rules 
            self.A = A                                  # our number of antecedents

            NoSamples = TrainData.shape[0]
                      
            # run the k-means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', tol=1e-6, max_iter=500, random_state=0).fit(TrainData)
            #print('K-means mean guess')
            #print(kmeans.cluster_centers_)
            # take the centers as our ant's
            mu = torch.rand(R,A) 
            for i in range(R): 
                for j in range(A):
                    mu[i,j] = kmeans.cluster_centers_[i,j]
            self.mu = torch.nn.Parameter( mu )       
            
            # now, estimate the variances
            sig = torch.rand(R,A)
            for r in range(R):
                inds = np.where( kmeans.labels_ == r )  
                classdata = torch.squeeze( TrainData[inds,:] )
                for d in range(A):
                    sig[r,d] = torch.std( torch.squeeze(classdata[:,d]) )
            #print('K-means sigma guess')
            #print(sig)
            self.sigma = torch.nn.Parameter( sig )
            
            # just make random rho's
            self.rho = torch.nn.Parameter( torch.rand(R,A+1) ) # again, could be any +/-
            
            return
        
        elif( InitMethod == 2 ): # k-means based init with rho guess       

            self.R = R                                  # our number of rules 
            self.A = A                                  # our number of antecedents

            NoSamples = TrainData.shape[0]
                        
            # run the k-means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', tol=1e-6, max_iter=500, random_state=0).fit(TrainData)
            print('K-means mean guess')
            print(kmeans.cluster_centers_)
            # steal the cluster centers for our ant's
            mu = torch.rand(R,A) 
            for i in range(R): 
                for j in range(A):
                    mu[i,j] = kmeans.cluster_centers_[i,j]
            self.mu = torch.nn.Parameter( mu )       
            
            # now, estimate the variances
            sig = torch.rand(R,A)
            for r in range(R):
                inds = np.where( kmeans.labels_ == r )  
                classdata = torch.squeeze( TrainData[inds,:] )
                for d in range(A):
                    sig[r,d] = torch.std( torch.squeeze(classdata[:,d]) )
            print('K-means sigma guess')
            print(sig)
            self.sigma = torch.nn.Parameter( sig )
            
            # now, guess at rhos, using least means squared
            
            # first, calc the constants (since we now have fixed means and std's from above)
            # yes, this could be wrote a lot nicer!!! (less code and more pretty)
            Ws = torch.zeros(NoSamples,R,A)
            Wprods = torch.ones(NoSamples,R)
            Wsums = torch.zeros(NoSamples) + 0.00000001 
            for s in range(NoSamples):
                for i in range(R):
                    for j in range(A):
                        Ws[s,i,j] = math.exp( - ((TrainData[s,j]-mu[i,j])**2) / (2*sig[i,j]**2) )
                        Wprods[s,i] = Wprods[s,i] * Ws[s,i,j] # each rule is the t-norm (product here) of the MFs
                    Wsums[s] = Wsums[s] + Wprods[s,i] # we will normalize by this below
                    
            # make up our matrix to solve for
            CoeffMatrix = np.zeros((NoSamples,R*(A+1)))
            for s in range(NoSamples):
                ctr = 0
                for i in range(R):
                    for j in range(A):
                        CoeffMatrix[s,ctr] = Wprods[s,i] * TrainData[s,j] * ( 1.0 / Wsums[s] )
                        ctr = ctr + 1
                    # now, do for bias term
                    CoeffMatrix[s,ctr] = Wprods[s,i] * 1.0 * ( 1.0 / Wsums[s] )
                    ctr = ctr + 1
                    
            # now, solve for rho
            p, res, rnk, s = lstsq(CoeffMatrix, TrainLabels)
            
            # now, format and return this as our init
            rho = torch.zeros(R,A+1)
            ctr = 0
            for i in range(R):
                for j in range(A+1):
                    rho[i,j] = p[ctr]
                    ctr = ctr + 1
            print('Rho guess is')
            print(rho)
            self.rho = torch.nn.Parameter( rho ) 
            
            return            
            
# Now, lets wrap the above ANFIS neuron into a layer so one can make a deep ANFIS
# the below are examples of moving an ANFIS neuron into different "types" of layers
        
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

    # init function that makes layers
    def __init__(self, N_in, N_out, N_rules, InitMethods=0):
        
        super(FuzzyNeuronLayer,self).__init__()
        
        self.N_out = N_out 
        
        if (isinstance(N_rules, int)):
            N_rules = [N_rules for i in range(self.N_out)]
            
        if (isinstance(InitMethods, int)):
            InitMethods = [InitMethods for i in range(self.N_out)]
            
        if self.N_out>1: 
            self.fuzzyneurons = torch.nn.ModuleList([FuzzyNeuron(N_rules[i], N_in, InitMethods[i]) for i in range(self.N_out)])
        else: 
            self.fuzzyneurons = FuzzyNeuron(N_rules, N_in, InitMethods)
       
    def forward(self,input):
        
        if self.N_out>1:
            out = torch.stack([fn(input) for fn in self.fuzzyneurons], dim=1)
        else:
            out = self.fuzzyneurons(input)
        
        return out

# OK, here are examples of how to use the above code
# If you run 'python ANFISPyTorchDeep.py' the below is executed

if __name__=='__main__':

    # In this example, we show how to make multiple layers (aka you can achieve a "deep ANFIS")
    if(0): 
        
        # An example of two layer ANFIS with 4 inputs, 3 hidden neurons and 2 outputs
        class TwoLayersANFIS(torch.nn.Module):
        
            # init function that makes layers
            def __init__(self):
                
                super(TwoLayersANFIS,self).__init__()  
                self.two_layer_anfis = torch.nn.Sequential(
                        # First layer has 4 inputs and 3 outputs (aka ANfIS neurons)
                        # The number of rules are 2, 3, and 2 respectively
                        FuzzyNeuronLayer(4,3,N_rules=[2,3,2]),
                        # Second layer with 3 inputs and two outputs. All neurons have 2 rules.
                        FuzzyNeuronLayer(3,2,N_rules=2)
                        )
             
            def forward(self,input):
                return self.two_layer_anfis(input)
    
        net =  TwoLayersANFIS()                 # initialize two layer ANFIS net
        data = torch.rand(7,4)                  # input dataset with 7 samples and 4 attributes         
        output = net(data)                      # forward pass
        
    # Case where we make synthetic data, a single ANFIS, and see if we can learn it    
    else:

        ###################################################
        ###################################################
        ###################################################
        ###################################################

        # define a Gaussian for below code
        def MF(x, mu, sigma):
            return math.exp( - ((x-mu)**2) / (2*sigma**2) )

        # ----------------------------------------
        # data -----------------------------------
        NoSamples = 300
        NoPatterns = 3 # how many rules do we want to make?
        x = torch.zeros(NoSamples,2)

        # ----------------------------------------
        # labels ---------------------------------
        l = torch.zeros(NoSamples)
        # randomly pick the model parameters
        Membs = torch.rand(NoPatterns,2) # the means (for the membership functions)
        Stds = torch.rand(NoPatterns,2)*.1 # the standard deviations (for the membership functions)
        Rows = torch.rand(NoPatterns,3) # these are the linear rules weights (the consequence part of our rules)
        # sample some random data (aka, make our NoPatterns of rules)
        dc = 0
        for r in range(NoPatterns):
            tm = torch.Tensor([Membs[r,0], Membs[r,1]]) 
            tc = torch.eye(2)
            tc[0,0] = Stds[r,0]*Stds[r,0]
            tc[1,1] = Stds[r,1]*Stds[r,1]
            m = torch.distributions.multivariate_normal.MultivariateNormal(tm,tc)
            for i in range(int(NoSamples/NoPatterns)):
                x[dc,:] = m.sample()
                dc = dc + 1
        # now, have to fire each rule
        m = torch.rand(NoSamples)
        ll = torch.rand(NoSamples)
        for i in range(NoSamples):
            mm = 0
            for r in range(NoPatterns):
                m[r] = MF(x[i,0], Membs[r,0], Stds[r,0]) * MF(x[i,1], Membs[r,1], Stds[r,1])
                mm = mm + m[r]
                ll[r] = x[i,0] * Rows[r,0] + x[i,1] * Rows[r,1] + Rows[r,2]
            for r in range(NoPatterns):
                l[i] = l[i] + ll[r]*m[r]
            l[i] = l[i] / (mm + 0.0000000000001)
        print('Done with building synthetic data set')

        ###################################################
        ###################################################
        ###################################################
        ###################################################

        # train now !!! --------------------------

        # make up training data set
        dataset = {'samples': x, 'labels': l} 
        train = Format_Dataset(dataset, choice = 'Train')
        train = torch.utils.data.DataLoader( shuffle = True,
                                                 dataset = train, 
                                                 batch_size = 15 ) 
                                                 
        R = 3 # 2 rules	
        A = 2 # 2 antecedents
        net = FuzzyNeuron(R,A,2,x,l)

        # set up optimization parameters
        criterion = torch.nn.MSELoss()
        #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.3)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        NoEpochs = 100
        for epoch in tqdm(range( NoEpochs ),'Training Epoch'):
            for sample, label in train:
                outputs = net(sample)
                loss = criterion(outputs, label)        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()            
        print('Done learning')
        
        ###################################################
        ###################################################
        ###################################################
        ###################################################

        # its in 2D, lets plot!
        #plt.figure()
        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        data = np.asarray(x)
        MU = (net.mu.data).cpu().numpy()
        SIGMA = (net.sigma.data).cpu().numpy()
        colormap = ['g', 'k', 'c', 'g', 'k', 'c']
        for r in range(NoPatterns):
            for d in range( int(NoSamples/NoPatterns) ):
                plt.plot( data[(int(NoSamples/NoPatterns))*r+d,0] , data[(int(NoSamples/NoPatterns))*r+d,1] , '.', color=colormap[r] )
        # show rule coverage
        for r in range(R):
            for a in range(3):
                ellipse = Ellipse(xy=(MU[r,0], MU[r,1]), width=(a+1)*SIGMA[r,0], height=(a+1)*SIGMA[r,1], 
                            edgecolor='r', fc='None', lw=(a+1))
                ax.add_patch(ellipse)
            plt.plot( MU[r,0] , MU[r,1] , 'bo' )     
        plt.show()