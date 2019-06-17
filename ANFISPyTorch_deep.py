# University of Missouri
#
# POC
#       Derek Anderson
#       AndersonDT@missouri.edu
# Last Date Modified
#       5-19-2019
# Description
#       Initial PyTorch code for first order TSK fuzzy rule base neuron
# Assumptions
#       Use's Gaussian membership OR {0,1} trap functions
#       Use's fixed # of rules and # of antecedents (getting away from that next)
#       Support serial processing versus mini-batch (need to add)
#       Uses PyTorch autograd for backprop (we worked out derivatives fyi)
#       Need to speed this code up

import torch
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import lstsq
from sklearn.cluster import DBSCAN
import sys
import argparse
import yaml

# FUZZY INFERENCE SYSTEM NEURON ------------------------------------------------

class FuzzyNeuron(torch.nn.Module):

    ############################################
    ############################################
    ############################################
    ############################################
    def forward(self,input):
        """ forward pass """

        # wow, love PyTorch, write in torch commands, get those back prop grad's for free! ;-)

        z = torch.zeros( input.shape[0], self.R )    # our rule outputs
        w = torch.ones( self.R, input.shape[0] )     # our weights

        for k in range( input.shape[0] ):            # go over all samples (make lin algebra in future!)
            for r in range( self.R ):                # do for each rule (make lin algebra in future!)
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
    def __init__(self, R, A, params, TrainData=[], BinaryLabels=[]): # smarter init

        super(FuzzyNeuron,self).__init__()              # call parent function

        # get parameters
        UseLOSNAsRules = params["ANFISUseLOSNAsRules"]  # use perceptron or LOSN for rules?
        WhatMF = params["ANFISWhatMF"]                  # what membership function to use?
        SmartInit = params["ANFISSmartInitMethod"]      # do random or smart initilization?
        ANFISOptimizeMFs = params["ANFISOptimizeMFs"]   # do we want to optimize the membership functions?
        ANFISGuessRho = params["ANFISGuessRho"]         # do we guess at rho's? (e.g., LMS)

        # store some parameters that we will need to know later (e.g., if/when implemented in the forward pass)
        self.WhatMF = WhatMF
        self.UseLOSNAsRules = UseLOSNAsRules

        # random (ish) init ---------------------------------------------------------
        if( SmartInit == 0 ): # random init

            # sorry, whatever you wanted, random init, lets optimize all things!!!
            # user might of entered some wacky combo of parameters, I am overriding

            self.R = R                                  # our number of rules
            self.A = A                                  # our number of antecedents

            self.mu = torch.nn.Parameter( torch.rand(R,A) )    # in [0,1], but could be +/-, whatever
            self.sigma = torch.nn.Parameter( torch.ones(R,A) * 0.2 ) # don't get's around 0 (semantically)
            # self.rho = torch.nn.Parameter( torch.rand(R,A+1) ) # again, could be any +/-
            self.rho = torch.nn.Parameter( torch.zeros(R,A+1) ) # again, could be any +/-

            return

        # k-means init ------------------------------------------------------------
        elif( SmartInit == 1 ):

            # here, I only allow
            #   Gaussian MF
            #   learning the Gauss MF or keeping it fixed
            #   learning the rho using LMS
            # no other combos; e.g., LOSN, trap MF, etc.

            self.R = R                                  # our number of rules (in future, learn!)
            self.A = A                                  # our number of antecedents (learn later!)

            # how much training data did we get?
            NoSamples = TrainData.shape[0]

            # a fixed "small" sigma (not too small to starve us, not to big to cover everything! means what?...)
            # in the future, we can mine this from the clustering results, this is just stab one (a constant)
            sigval = 0.3
            if(ANFISOptimizeMFs == 1):
                self.sigma = torch.nn.Parameter( torch.ones(R,A) * sigval )
            else:
                self.sigma = torch.ones(R,A) * sigval

            # for R rules, we will find R clusters - yes, code is ugly, someone can clean this up!
            kmeans = KMeans(n_clusters=R, random_state=0).fit(TrainData)
            mu = torch.rand(R,A)
            for i in range(R):
                for j in range(A):
                    mu[i,j] = kmeans.cluster_centers_[i,j]
            if(ANFISOptimizeMFs == 1):
                self.mu = torch.nn.Parameter( mu )
            else:
                self.mu = mu

            # now, solve for rho

            if(ANFISGuessRho==1):

                # first, calc the constants (since we now have fixed means and std's)
                Ws = torch.zeros(NoSamples,R,A)
                Wprods = torch.ones(NoSamples,R)
                Wsums = torch.zeros(NoSamples) + 0.00000001 # add some pad
                for s in range(NoSamples):
                    for i in range(R):
                        for j in range(A):
                            Ws[s,i,j] = math.exp( - ((TrainData[s,j]-mu[i,j])**2) / (2*sigval**2) )
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
                p, res, rnk, s = lstsq(CoeffMatrix, BinaryLabels)

                # now, format and return this as our init
                rho = torch.zeros(R,A+1)
                ctr = 0
                for i in range(R):
                    for j in range(A+1):
                        rho[i,j] = p[ctr]
                        ctr = ctr + 1
                self.rho = torch.nn.Parameter( rho )

            else:

                self.rho = torch.nn.Parameter( torch.rand(R,A+1) ) # again, could be any +/-
                # self.rho = torch.nn.Parameter( torch.zeros(R,A+1) ) # again, could be any +/-

            return

        # DBSCAN init ------------------------------------------------------------
        elif( SmartInit == 2 ):

            # here, I only allow:
            #  Gaussian MF
            #  keep Gaussian fixed or can learn in
            #  estimate rho using the LMS
            # no other combos; e.g., LOSN, trap MF, etc.

            # how much training data did we get?
            NoSamples = TrainData.shape[0]

            # cluster
            clustering = DBSCAN(eps=0.1, min_samples=5).fit( TrainData )

            # how many clusters?
            NC = np.amax(clustering.labels_) + 1

            # get the -1's (the outliers...)
            Outliers = np.where(clustering.labels_ == -1)[0]

            # determine the # of rules
            R = Outliers.size + NC
            self.R = R
            self.A = A

            # make each cluster
            mu = torch.zeros(R,A)
            counters = np.zeros(R)
            for i in range( NC ):
                for k in range(NoSamples):
                    if( clustering.labels_[k] == i ):
                        for j in range(A):
                            mu[i,j] = mu[i,j] + TrainData[k,j]
                        counters[i] = counters[i] + 1
            for i in range( NC ):
                mu[i,:] = mu[i,:] / counters[i]
            # now, do special cases/exceptions
            incx = NC
            for k in range(NoSamples):
                if( clustering.labels_[k] == -1 ):
                    for j in range(A):
                        mu[incx,j] = TrainData[k,j]
                    incx = incx + 1
            #now store it
            if(ANFISOptimizeMFs == 1):
                self.mu = torch.nn.Parameter( mu )
            else:
                self.mu = mu

            # make sigma
            sigval = torch.zeros(R,A)
            for i in range( R ):
                if( i < NC ):
                    for k in range(NoSamples):
                        if( clustering.labels_[k] == i ):
                            for j in range(A):
                                sigval[i,j] = sigval[i,j] + ( TrainData[k,j] - mu[i,j] )*( TrainData[k,j] - mu[i,j] )
                    for j in range(A):
                        sigval[i,j] = (sigval[i,j] / counters[i]) + 0.01
                else:
                    for j in range(A):
                        sigval[i,j] = 0.1
            # now store it
            if(ANFISOptimizeMFs == 1):
                self.sigma = torch.nn.Parameter( sigval )
            else:
                self.sigma = sigval

            # now, solve for rho

            if(ANFISGuessRho==1):

                # first, calc the constants (since we now have fixed means and std's)
                Ws = torch.zeros(NoSamples,R,A)
                Wprods = torch.ones(NoSamples,R)
                Wsums = torch.zeros(NoSamples) + 0.00000001 # add some pad
                for s in range(NoSamples):
                    for i in range(R):
                        for j in range(A):
                            Ws[s,i,j] = math.exp( - ((TrainData[s,j]-mu[i,j])**2) / (2*sigval[i,j]**2) )
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
                p, res, rnk, s = lstsq(CoeffMatrix, BinaryLabels)

                # now, format and return this as our init
                rho = torch.zeros(R,A+1)
                ctr = 0
                for i in range(R):
                    for j in range(A+1):
                        rho[i,j] = p[ctr]
                        ctr = ctr + 1
                self.rho = torch.nn.Parameter( rho )

            else:

                self.rho = torch.nn.Parameter( torch.rand(R,A+1) ) # again, could be any +/-
                # self.rho = torch.nn.Parameter( torch.zeros(R,A+1) ) # again, could be any +/-

            return

        # zero init --------------------------------------------------------------
        elif( SmartInit == 3 ):

            self.R = R                                  # our number of rules
            self.A = A                                  # our number of antecedents

            if(ANFISOptimizeMFs == 1):
               self.mu = torch.nn.Parameter( torch.zeros(R,A) )
               self.sigma = torch.nn.Parameter( torch.zeros(R,A) )
            else:
               self.mu = torch.zeros(R,A)
               self.sigma = torch.zeros(R,A)
            self.rho = torch.nn.Parameter( torch.zeros(R,A+1) ) # again, could be any +/-

            return


class FuzzyNeuronLayer(torch.nn.Module):

    def __init__(self, R_in, A_in, paramsList, TrainData=[], BinaryLabels=[], N_out=1):
        super(FuzzyNeuronLayer,self).__init__()
        self.N_out = N_out
        if self.N_out>1:
            self.fuzzyneurons = torch.nn.ModuleList([FuzzyNeuron(R_in[i], A_in[i], paramsList[i], TrainData=TrainData, BinaryLabels=BinaryLabels) for i in range(self.N_out)])
        else:
            self.fuzzyneurons = FuzzyNeuron(R_in, A_in, paramsList, TrainData=TrainData, BinaryLabels=BinaryLabels)


    def forward(self,inputs):
        if self.N_out>1:
            out = torch.cat([fn(inputs) for fn in self.fuzzyneurons], dim=0)
        else:
            out = self.fuzzyneurons(inputs)
        return out

if __name__=='__main__':
    R_in = [5,4]
    A_in = [3,6]
    N_out = 2
    paramsList = []
    params = {}
    params["ANFISUseLOSNAsRules"] = 0       # use perceptron or LOSN for rules?
    params["ANFISWhatMF"] = 0               # what membership function to use?
    params["ANFISSmartInitMethod"] = 0      # do random or smart initilization?
    params["ANFISOptimizeMFs"] = 0          # do we want to optimize the membership functions?
    params["ANFISGuessRho"] = 0
    paramsList.append(params)
    paramsList.append(params)


    net = FuzzyNeuronLayer(R_in,A_in, paramsList)
