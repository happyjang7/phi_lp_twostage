from __future__ import print_function
import numpy as np
import sys, os
sys.path.append("/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
import cplex
import lp, PhiDivergence
import scipy.io as sio
from scipy.sparse import csr_matrix, find, coo_matrix, hstack, vstack

class set(object):
    def __init__(self, inLPModel, inPhi, inNumObsPerScen, inCutType='multi'):
        self.lp = inLPModel
        self.phi = inPhi;
        self.obs = inNumObsPerScen
        self.cutType = inCutType

        self.MASTER = 0
        self.TRUE = 1

        self.SLOPE = 0
        self.INTERCEPT = 1

        self.numVariables = self.lp.first_obj.size
        self.numScen = self.lp.numScenarios
        self.phiLimit = np.minimum(self.phi.limit(), self.phi.computationLimit)
        self.isObserved = self.obs > 0

        if inCutType == "single":
            self.numTheta = 1
        elif inCutType == "multi":
            self.numTheta = self.numScen
        else:
            raise Exception('Cut types available: single and multi')

        self.Reset()

    def Reset(self):
        self.solution = np.zeros_like(self.lp.first_obj,dtype=float)
        self.lambda1 =np.float64(1)
        self.mu = np.float64(0)

        self.theta = [-cplex.infinity*np.ones(self.numTheta), -cplex.infinity*np.ones(self.numTheta)]

        self.secondStageValues = -cplex.infinity*np.ones(self.numScen)
        self.secondStageDuals = [[np.array([]) for _ in range(self.numScen)], [np.array([]) for _ in range(self.numScen)]]
        self.secondStageSolutions = [np.array([]) for _ in range(self.numScen)]

        self.muFeasible = np.nan



    def SetX(self, inX):
        if np.array(inX).size != self.numVariables:
            raise Exception('X has size '+str(np.array(inX).size)+', should be '+str(self.numVariables))
        self.solution = inX

    def SetLambda(self, inLambda):
        if np.array(inLambda).size != 1:
            raise Exception('Lambda has size '+str(np.array(inLambda).size)+', should be 1')
        elif inLambda < 0:
            raise Exception('Lambda must be non-negative')
        self.lambda1 = inLambda

    def SetMu(self, inMu):
        if np.array(inMu).size != 1:
            raise Exception('Mu has size '+str(np.array(inMu).size)+', should be 1')
        self.mu = inMu

    def SetTheta(self, inTheta, inType):
        if np.array(inTheta).size != self.theta[0].size:
            raise Exception('Theta has size '+str(np.array(inTheta).size)+', should be '+str(self.numScen))
        if inType == 'master':
            typeN = self.MASTER
        elif inType == 'true':
            typeN = self.TRUE
        else:
            raise Exception('type must be ''master'' or ''true''')
        self.theta[typeN] = inTheta

    def SetSecondStageValue(self, inScen, inValue):
        if inScen < 0 or inScen > self.numScen-1:
            raise Exception('Scenario number must be between 0 and '+str(self.numScen-1))
        self.secondStageValues[inScen] = inValue

        if np.all(self.secondStageValues > -cplex.infinity):
            tolerBound = np.float64(1e-6) * np.maximum(self.phiLimit, 1)
            if tolerBound == np.inf:
                tolerBound = cplex.infinity
            self.muFeasible = np.all(self.S() < self.phiLimit + (~self.isObserved * 1.0) * tolerBound)

    def SetSecondStageDual(self, inScen, inDual, inType):
        if inScen < 0 or inScen > self.numScen-1:
            raise Exception('Scenario number must be between 0 and '+str(self.numScen-1))
        if inType == 'slope':
            type = self.SLOPE
        elif inType == 'int':
            type = self.INTERCEPT
        else:
            raise Exception('type must be ''slope'' or ''int''')
        self.secondStageDuals[type][inScen] = inDual

    def SetSecondStageSolution(self, inScen, inSol):
        if inScen < 0 or inScen > self.numScen-1:
            raise Exception('Scenario number must be between 0 and '+str(self.numScen-1))
        self.secondStageSolutions[inScen] = inSol

    def X(self):
        return self.solution

    def Lambda(self):
        return self.lambda1

    def Mu(self):
        return self.mu

    def ThetaMaster(self):
        return self.theta[self.MASTER]

    def ThetaTrue(self):
        return self.theta[self.TRUE]

    def S(self):
        if self.lambda1 != 0:
            return (self.secondStageValues - self.mu) / self.lambda1
        else:
            relDiff = (self.secondStageValues - self.mu) / np.abs(self.mu)
            outS = np.zeros_like(self.secondStageValues,dtype=float)
            tol = np.float64(1e-6)
            outS[relDiff < -tol] = -cplex.infinity
            outS[relDiff >  tol] = cplex.infinity
            return outS

    def Limit(self):
        return self.phiLimit

    def SecondStageValues(self):
        return self.secondStageValues

    def MuFeasible(self):
        return self.muFeasible

    def SecondStageSlope(self, inScen):
        return self.secondStageDuals[self.SLOPE][inScen]

    def SecondStageIntercept(self, inScen):
        return self.secondStageDuals[self.INTERCEPT][inScen]

    def SecondStageSolution(self, inScen):
        return self.secondStageSolutions[inScen]


if __name__ == "__main__":
    mat_data = sio.loadmat(os.getcwd() + "/mat_data/apl1p6.mat")
    lp = lp.set(mat_data)
    inPhi = PhiDivergence.set('mchi2')
    obs   = np.array([1,1,1,1,1,1])
    inRho = inPhi.Rho(0.05, obs)

    solution=set(lp, inPhi,obs)

