'''
PhiLP solves a 2 Stage ﻿Data-Driven Stochastic Programming Using Phi-Divergences with Recourse (PhiLP-2)
via the modified Bender's Decomposition proposed by David Love.
This class uses the lp.set class to create and store the LP data.
required modules
1. numpy, scipy.stats
2. warnings, copy
3. cplex, cplex.exceptions, cplex.callbacks
4. lp, ExtensiveForm, PhiDivergence, Solution
'''
from __future__ import print_function
import sys
import os
import scipy.io as sio
from scipy.sparse import csr_matrix, find, coo_matrix, hstack, vstack

sys.path.append("/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
import numpy as np
from scipy.stats import chi2
import warnings, copy
import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import SimplexCallback
import lp, ExtensiveForm, ExtepectedForm, PhiDivergence, Solution
import copy

class MyCallback(SimplexCallback):
    def __call__(self):
        print("CB Iteration ", self.get_num_iterations(), " : ", end=' ')
        if self.is_primal_feasible():
            print("CB Objective = ", self.get_objective_value())
        else:
            print("CB Infeasibility measure = ",
                  self.get_primal_infeasibility())


class set(object):
    def __init__(self, inLPModel, inPhi, inNumObsPerScen, inRho, inOptimizer='cplex', inCutType='multi'):
        self.lpModel = inLPModel
        self.phi = inPhi
        self.numObsPerScen = np.array(inNumObsPerScen)
        self.numObsTotal = np.sum(self.numObsPerScen)

        if inRho < 0:
            phi2deriv = np.float32(self.phi.SecondDerivativeAt1())
            if np.isfinite(phi2deriv) and phi2deriv > 0:
                inRho = (phi2deriv / (2 * self.numObsTotal)) * chi2.ppf(0.95, self.lpModel.numScenarios - 1)
            else:
                raise Exception('Second derivative of phi(t) does not allow for automatically setting rho')

        if self.lpModel.numStages != 2:
            raise Exception('Must use a 2 Stage LP')
        if len(inNumObsPerScen) != self.lpModel.numScenarios:
            raise Exception('Size of observations differs from number of scenarios')

        self.rho = inRho
        self.phi.SetComputationLimit(self.numObsPerScen, self.rho)

        self.optimizer = inOptimizer

        self.objectiveTolerance = np.float32(1e-6)
        self.probabilityTolerance = np.float32(5e-3)

        self.LAMBDA = self.lpModel.first_obj.size
        self.MU = self.LAMBDA + 1

        if inCutType == "single":
            thetaOffset = 1
        elif inCutType == "multi":
            thetaOffset = np.arange(1, self.lpModel.numScenarios + 1)
        else:
            raise Exception('Cut types available: single and multi')
        self.THETA = self.MU + thetaOffset

        self.SLOPE = 0
        self.INTERCEPT = 1

        self.candidateSolution = Solution.set(self.lpModel, self.phi, self.numObsPerScen, inCutType='multi')
        self.InitializeBenders()

    def InitializeBenders(self):
        self.objectiveCutsMatrix = np.array([])
        self.objectiveCutsRHS = np.array([])
        self.feasibilityCutsMatrix = np.array([])
        self.feasibilityCutsRHS = np.array([])
        self.zLower = -np.inf
        self.zUpper = np.inf

        self.bestSolution = np.array([])
        self.secondBestSolution = np.array([])

        self.ResetSecondStageSolutions()
        self.newSolutionAccepted = True
        x0, exitFlag, slack, pi, dj = ExtepectedForm.solve(self.lpModel)
        # x0, exitFlag = ExtensiveForm.solve(self.lpModel)
        if exitFlag != 1:
            raise Exception('Could not solve first stage LP')
        cols = self.lpModel.first_obj.size
        x0 = x0[0:cols]

        self.lambdaLowerBound = np.float64(1e-6)
        self.objectiveScale = np.float64(1)

        self.candidateSolution.SetX(x0)
        self.candidateSolution.SetLambda(np.float64(1))
        self.candidateSolution.SetMu(np.float64(0))
        self.SolveSubProblems()
        self.GenerateCuts()
        self.UpdateBestSolution()
        self.UpdateTolerances()

    def SolveMasterProblem(self):
        self.ResetSecondStageSolutions()

        cMaster = self.GetMasterc()
        AMaster = self.GetMasterA()
        bMaster = self.GetMasterb()
        lMaster = self.GetMasterl()
        uMaster = self.GetMasteru()
        senseMaster = self.GetMastersense()

        CutMatrix = np.vstack((self.objectiveCutsMatrix, self.feasibilityCutsMatrix))
        rows, cols = CutMatrix.nonzero()
        idx = list(zip(rows, cols))
        CutMatrix_coefficients = [(int(rows[i] + bMaster.size), int(cols[i]), CutMatrix[idx[i]]) for i in
                                  range(len(idx))]
        CutMatrixRHS = np.hstack((self.objectiveCutsRHS, self.feasibilityCutsRHS))
        CutSense = ["L"] * CutMatrixRHS.size

        currentBest = self.GetDecisions(self.bestSolution)
        self.candidateSolution.Reset()
        try:
            mdl_master = cplex.Cplex()
            mdl_master.set_problem_name("mdl_master")
            mdl_master.parameters.lpmethod.set(mdl_master.parameters.lpmethod.values.auto)
            mdl_master.objective.set_sense(mdl_master.objective.sense.minimize)
            mdl_master.variables.add(obj=cMaster, lb=lMaster, ub=uMaster)
            mdl_master.linear_constraints.add(senses=senseMaster + CutSense, rhs=np.hstack((bMaster, CutMatrixRHS)))
            mdl_master.linear_constraints.set_coefficients(list(AMaster) + CutMatrix_coefficients)

            mdl_master.set_results_stream(None)
            mdl_master.solve()
            mdl_master.register_callback(MyCallback)

            solution = mdl_master.solution
            numvars = mdl_master.variables.get_num()
            currentCandidate = np.array(solution.get_values(0, numvars - 1))
            exitFlag = solution.get_status()
        except CplexError as exc:
            print(exc)

        if currentCandidate[self.LAMBDA] < lMaster[self.LAMBDA]:
            if self.phi.Conjugate(-np.inf) == -np.inf:
                currentCandidate[self.LAMBDA] = lMaster[self.LAMBDA]
            elif currentCandidate[self.LAMBDA] < 0:
                currentCandidate[self.LAMBDA] = 0
            self.lambdaLowerBound = self.lambdaLowerBound * np.float64(1e-3)

        self.candidateSolution.SetX(currentCandidate[range(currentCandidate.size - 2 - self.THETA.size)])
        self.candidateSolution.SetLambda(currentCandidate[self.LAMBDA])
        self.candidateSolution.SetMu(currentCandidate[self.MU])
        self.candidateSolution.SetTheta(currentCandidate[self.THETA], 'master')

        if exitFlag != 1 or np.matmul(cMaster, currentBest - currentCandidate) < -1e-4 * np.matmul(cMaster,
                                                                                                   currentBest):
            if exitFlag == 1:
                print('Current Best solution value:', str(np.matmul(cMaster, currentBest)),
                      ', Candidate solution value:', str(np.matmul(cMaster, currentCandidate)))
                exitFlag = -50
            return

        if not (exitFlag != 1 or np.matmul(cMaster, currentBest - currentCandidate) >= -1e-4 * np.matmul(cMaster,
                                                                                                         currentBest)):
            raise Exception('Actual objective drop = ' + str(np.matmul(cMaster, (currentBest - currentCandidate))))

        return exitFlag

    def SolveSubProblems(self):
        solution = self.candidateSolution
        for scenarioNum in range(self.lpModel.numScenarios):
            self.SubProblem(scenarioNum, solution)

    def SubProblem(self, inScenNumber, inSolution):
        q = self.lpModel.second_obj[inScenNumber] * self.objectiveScale
        D = self.lpModel.second_coefficients[inScenNumber]
        d = self.lpModel.second_rhs[inScenNumber]
        B = self.lpModel.second_B[inScenNumber]
        l = self.lpModel.second_lb[inScenNumber]
        u = self.lpModel.second_ub[inScenNumber]
        sense = self.lpModel.second_sense[inScenNumber]

        xLocal = inSolution.X()
        try:
            mdl_sub = cplex.Cplex()
            mdl_sub.set_problem_name("mdl_sub")
            mdl_sub.parameters.lpmethod.set(mdl_sub.parameters.lpmethod.values.auto)

            mdl_sub.objective.set_sense(mdl_sub.objective.sense.minimize)
            mdl_sub.variables.add(obj=q, lb=l, ub=u)
            mdl_sub.linear_constraints.add(senses=sense, rhs=d + xLocal * B.transpose())
            mdl_sub.linear_constraints.set_coefficients(D)
            mdl_sub.set_results_stream(None)
            mdl_sub.solve()
            mdl_sub.register_callback(MyCallback)
            solution = mdl_sub.solution

            exitFlag = solution.get_status()
        except CplexError as exc:
            print(exc)
        if exitFlag != 1:
            warnings.warn("'***Scenario '" + str(inScenNumber) + "' exited with flag '" + str(exitFlag) + "'")

        y = np.array(solution.get_values())
        fval = np.float64(solution.get_objective_value())
        pi = np.array(solution.get_dual_values())
        dj = np.array(solution.get_reduced_costs())
        dj_l = copy.copy(dj)
        dj_l[dj_l < 0] = 0
        dj_u = copy.copy(dj)
        dj_u[dj_u > 0] = 0

        inSolution.SetSecondStageSolution(inScenNumber, y)
        inSolution.SetSecondStageDual(inScenNumber, np.transpose(pi) * B, 'slope')
        inSolution.SetSecondStageDual(inScenNumber, np.matmul(np.transpose(pi), d) +
                                      np.matmul(np.transpose(dj_u[u < cplex.infinity]), u[u < cplex.infinity]) -
                                      np.matmul(np.transpose(dj_l[l != 0]), l[l != 0]), 'int')
        inSolution.SetSecondStageValue(inScenNumber, fval)

    def GenerateCuts(self):
        if ~self.candidateSolution.MuFeasible():
            self.GenerateFeasibilityCut()
            self.FindFeasibleMu()
        self.FindExpectedSecondStage()
        self.GenerateObjectiveCut()

    def GenerateObjectiveCut(self):
        xLocal = self.candidateSolution.X()
        lambdaLocal = self.candidateSolution.Lambda()
        muLocal = self.candidateSolution.Mu()

        lambdaZero = False
        if lambdaLocal == 0:
            lambdaZero = True
            lower = self.GetMasterl()
            self.candidateSolution.SetLambda(lower[self.LAMBDA])
            lambdaLocal = self.candidateSolution.Lambda()

        s = self.candidateSolution.S()
        conjVals = self.phi.Conjugate(s)
        conjDerivs = self.phi.ConjugateDerivative(s)

        intermediateSlope = np.array([np.hstack((conjDerivs[ii] * self.candidateSolution.SecondStageSlope(ii),
                                                 np.array(self.rho + conjVals[ii] - conjDerivs[ii] * s[ii]),
                                                 np.array(1 - conjDerivs[ii]))) for ii in
                                      range(self.lpModel.numScenarios)])
        intermediateSlope[np.where(self.numObsPerScen == 0)[0]] = 0
        if self.THETA.size == 1:
            slope = np.matmul(self.numObsPerScen / self.numObsTotal, intermediateSlope)
        elif self.THETA.size == self.lpModel.numScenarios:
            slope = intermediateSlope
        else:
            raise Exception('Wrong size of obj.THETA.  This should not happen')

        intercept = self.candidateSolution.ThetaTrue() - np.matmul(slope, np.transpose(
            np.hstack((xLocal, lambdaLocal, muLocal))))

        if self.objectiveCutsMatrix.size == 0 and self.objectiveCutsRHS.size == 0:
            self.objectiveCutsMatrix = np.hstack((slope, -np.eye(self.THETA.size)))
            self.objectiveCutsRHS = -intercept
        else:
            self.objectiveCutsMatrix = np.vstack(
                [self.objectiveCutsMatrix, np.hstack((slope, -np.eye(self.THETA.size)))])
            self.objectiveCutsRHS = np.append(self.objectiveCutsRHS, -intercept)

        if lambdaZero:
            self.candidateSolution.SetLambda(0)

    def GenerateFeasibilityCut(self):
        hIndex = np.argmax(self.candidateSolution.SecondStageValues())
        limit = np.minimum(self.phi.limit(), self.phi.computationLimit)

        feasSlope = np.concatenate(
            [self.candidateSolution.SecondStageSlope(hIndex), -limit, np.array([-1]), np.zeros(self.THETA.size)])
        feasInt = self.candidateSolution.SecondStageIntercept(hIndex)

        if self.feasibilityCutsMatrix.size == 0 and self.feasibilityCutsRHS.size == 0:
            self.feasibilityCutsMatrix = np.array([feasSlope])
            self.feasibilityCutsRHS = -feasInt
        else:
            self.feasibilityCutsMatrix = np.vstack([self.feasibilityCutsMatrix, feasSlope])
            self.feasibilityCutsRHS = np.append(self.feasibilityCutsRHS, -feasInt)

    def FindFeasibleMu(self):
        lambdaLocal = self.candidateSolution.Lambda()
        limit = np.minimum(self.phi.limit(), self.phi.computationLimit)
        localValues = self.candidateSolution.SecondStageValues()
        mu = np.float64(np.max(localValues) - limit * np.float64(1 - 1e-3) * lambdaLocal)
        self.candidateSolution.SetMu(mu)

    def FindExpectedSecondStage(self):
        inSolution = self.candidateSolution
        if np.isnan(inSolution.MuFeasible()):
            raise Exception(
                'Must determine whether candidate mu is feasible before finding expected second stage value')
        if not all(inSolution.SecondStageValues() > -cplex.infinity):
            raise Exception('Must set second stage values before calculating expectation')
        lambdaLocal = inSolution.Lambda()
        muLocal = inSolution.Mu()

        rawTheta = muLocal + lambdaLocal * self.rho + lambdaLocal * self.phi.Conjugate(inSolution.S())
        rawTheta[np.where(self.numObsPerScen == 0)[0]] = 0
        rawTheta[np.where(rawTheta == cplex.infinity)[0]] = np.inf
        if not all(np.isreal(rawTheta)):
            raise Exception('Possible scaling error')
        if not all(np.isfinite(rawTheta)):
            raise Exception('Nonfinite theta, lambda = ' + str(lambdaLocal))

        if self.THETA.size == 1:
            inSolution.SetTheta(np.dot(self.numObsPerScen / self.numObsTotal, rawTheta), 'true')
        elif self.THETA.size == self.lpModel.numScenarios:
            inSolution.SetTheta(rawTheta, 'true')
        else:
            raise Exception('Wrong size of obj.THETA.  This should not happen')

    def UpdateSolutions(self):
        cMaster = self.GetMasterc()
        if np.matmul(cMaster, self.GetDecisions(self.candidateSolution, 'true')) < self.zUpper:
            self.newSolutionAccepted = True

        if self.newSolutionAccepted:
            self.UpdateBestSolution()
            bestDecisions = self.GetDecisions(self.bestSolution, 'true')
            self.zUpper = np.matmul(cMaster, bestDecisions)

        if self.candidateSolution.MuFeasible():
            candidateDecisions = self.GetDecisions(self.candidateSolution, 'master')
            self.zLowerUpdated = True
            self.zLower = np.matmul(cMaster, candidateDecisions)
        else:
            self.zLowerUpdated = False

    def UpdateTolerances(self):
        if self.zLower > -np.inf:
            self.currentObjectiveTolerance = (self.zUpper - self.zLower) / np.minimum(np.abs(self.zUpper),
                                                                                      np.abs(self.zLower))
        else:
            self.currentObjectiveTolerance = -np.inf
        self.currentProbabilityTolerance = np.abs(1 - np.sum(self.pWorst))

    def WriteProgress(self):
        print("=" * 100)
        print(self.phi.divergence, ', rho = ', str(self.rho))
        print('Observations: ', str(self.numObsPerScen))
        print(str(self.NumObjectiveCuts()), ' objective cuts, ', str(self.NumFeasibilityCuts()), ' feasibility cuts.')

        if self.candidateSolution.MuFeasible():
            print('No feasibility cut generated')
        else:
            print('Feasibility cut generated')

        if self.newSolutionAccepted:
            print('New solution, zupper = ', str(self.zUpper))
        else:
            print('No new solution accepted')

        if self.zLowerUpdated:
            print('New lower bound, zlower = ', str(self.zLower))
        else:
            print('No new lower bound found')
        print('Objective tolerance ', str(self.currentObjectiveTolerance))
        print('Probability tolerance ', str(self.currentProbabilityTolerance))

    def UpdateBestSolution(self):
        if self.newSolutionAccepted:

            if not np.array_equal(self.bestSolution, []):
                self.secondBestSolution = copy.copy(self.bestSolution)
            self.bestSolution = copy.copy(self.candidateSolution)
            self.CalculateProbability()

    def ForceAcceptSolution(self):
        self.newSolutionAccepted = True
        self.UpdateBestSolution()

    def CalculateProbability(self):
        q = self.numObsPerScen / self.numObsTotal
        s = self.bestSolution.S()
        self.pWorst = np.multiply(q, self.phi.ConjugateDerivative(s))
        self.pWorst[np.where(q == 0)[0]] = 0
        limitCases = np.abs(s - self.phi.limit()) <= np.float64(1e-6)
        if np.count_nonzero(limitCases) > 0:
            self.pWorst[limitCases] = (1 - np.sum(self.pWorst[np.logical_not(limitCases)])) / np.count_nonzero(
                limitCases)
            # self.pWorst[limitCases] = (2 * q[limitCases] + self.rho - np.sqrt((2 * q[limitCases] * self.rho) ** 2 - 4 * q[limitCases] ** 2)) / 2
        self.calculatedDivergence = np.sum(self.phi.Contribution(self.pWorst, q))

    def UpdateTolerances(self):
        if self.zLower > -np.inf:
            self.currentObjectiveTolerance = (self.zUpper - self.zLower) / np.minimum(np.abs(self.zUpper),
                                                                                      np.abs(self.zLower))
        else:
            self.currentObjectiveTolerance = np.inf
        self.currentProbabilityTolerance = np.abs(1 - np.sum(self.pWorst))

    def ResetSecondStageSolutions(self):
        self.candidateSolution.Reset()
        self.newSolutionAccepted = False
        self.zLowerUpdated = False

    def GetMasterc(self):
        cOut = np.append(self.lpModel.first_obj, np.zeros(2 + self.THETA.size)) * self.objectiveScale
        cOut[self.LAMBDA] = 0
        cOut[self.MU] = 0
        if self.THETA.size == 1:
            cOut[self.THETA] = 1;
        elif self.THETA.size == self.lpModel.numScenarios:
            cOut[self.THETA] = self.numObsPerScen / self.numObsTotal
        else:
            raise Exception('Wrong size of obj.THETA.  This should not happen')
        return cOut

    def GetMasterA(self):
        AOut = self.lpModel.first_coefficients
        return AOut

    def GetMasterb(self):
        bOut = self.lpModel.first_rhs
        return bOut

    def GetMasterl(self):
        lOut = np.append(self.lpModel.first_lb, np.zeros(2 + self.THETA.size))
        lOut[self.LAMBDA] = self.lambdaLowerBound
        lOut[self.MU] = -cplex.infinity
        lOut[self.THETA] = np.float64(-10 ** 19)
        return lOut

    def GetMasteru(self):
        uOut = np.append(self.lpModel.first_ub, np.zeros(2 + self.THETA.size))
        uOut[self.LAMBDA] = cplex.infinity
        uOut[self.MU] = cplex.infinity
        uOut[self.THETA] = cplex.infinity
        return uOut

    def GetMastersense(self):
        uOut = self.lpModel.first_sense
        return uOut

    def GetDecisions(self, solution, inType='true'):
        vOut = np.append(solution.X(), np.zeros(2 + self.THETA.size))
        vOut[self.LAMBDA] = solution.Lambda()
        vOut[self.MU] = solution.Mu()
        if inType == 'master':
            vOut[self.THETA] = solution.ThetaMaster()
        elif inType == 'true':
            vOut[self.THETA] = solution.ThetaTrue()
        else:
            raise Exception('Only accepts ''master and ''true''')
        return vOut

    def CandidateVector(self):
        outCV = self.GetDecisions(self.candidateSolution, 'master')
        return outCV

    def NumObjectiveCuts(self):
        outNum = self.objectiveCutsMatrix.shape[0]
        return outNum

    def NumFeasibilityCuts(self):
        outNum = self.feasibilityCutsMatrix.shape[0]
        return outNum

    def ObjectiveValue(self):
        outValue = self.zUpper
        return outValue

    def DoubleIterations(self):
        print('Max Iterations, not available to the Cplex solver ', )

    def DeleteOldestCut(self):
        self.objectiveCutsMatrix = self.objectiveCutsMatrix[self.THETA.size:]
        self.objectiveCutsRHS = self.objectiveCutsRHS[self.THETA.size:]

    def DeleteOldestFeasibilityCut(self):
        self.feasibilityCutsMatrix = self.feasibilityCutsMatrix[1:]
        self.feasibilityCutsRHS = self.feasibilityCutsRHS[1:]


if __name__ == "__main__":
    mat_data = sio.loadmat(os.getcwd() + "/mat_data/apl1p6.mat")
    lp = lp.set(mat_data)
    inPhi = PhiDivergence.set('mchi2')
    obs = np.array([1, 1, 1, 1, 1, 1])
    inRho = inPhi.Rho(0.05, obs)

    philp = set(lp, inPhi, obs, inRho)

    print(philp.feasibilityCutsRHS)
    philp.DeleteOldestFeasibilityCut()
    print("====" * 50)
    print(philp.feasibilityCutsRHS)
