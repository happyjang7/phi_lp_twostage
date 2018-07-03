import numpy as np
import lp, PhiDivergence, PhiLP
import time
import os
import scipy.io as sio

mat_data = sio.loadmat(os.getcwd() + "/mat_data/NI48.mat")
lp = lp.set(mat_data)

start = time.clock()
alpha = 0.05
inPhi = PhiDivergence.set('burg')
obs = lp.obs
inRho = inPhi.Rho(alpha, obs)
philp = PhiLP.set(lp, inPhi, obs, inRho)
totalProblemsSolved = 1
totalCutsMade = 1

while not (
        philp.currentObjectiveTolerance <= philp.objectiveTolerance and philp.currentProbabilityTolerance <= philp.probabilityTolerance):
    if totalProblemsSolved >= 100:
        break
    totalProblemsSolved = totalProblemsSolved + 1
    exitFlag = philp.SolveMasterProblem()

    if ('cS' in locals() or 'cS' in globals()) and (np.array_equal(cS, philp.CandidateVector())):
        print(' ')
        print('Repeat Solution')
        exitFlag = -100

    if exitFlag != 1 and exitFlag != -100:
        print('exitFlag = ', str(exitFlag))
        if exitFlag == 0:
            philp.DoubleIterations()
        elif exitFlag in [-2, -3, -4, -5]:
            # Do nothing extra
            pass
        elif exitFlag == 5:
            # exitFlag = 5 indicates, for CPLEX, that an optimal solution was found found, but with scaling issues.
            # No additional actions will be taken.
            pass
        elif exitFlag == -50:
            # The optimizer failed to find a solution better than philp.bestSolution. This has been observed with cplexlp.
            pass
        elif exitFlag == -100:
            # The optimizer returned the same solution as it found the previous time around. This has been observed with cplexlp.
            pass
        else:
            raise Exception('Unknown error code: ' + str(exitFlag))
        if philp.NumObjectiveCuts() > philp.THETA.size:
            philp.DeleteOldestCut()
        if philp.NumFeasibilityCuts() > 1:
            philp.DeleteOldestFeasibilityCut()
        print(str(philp.NumObjectiveCuts()), ' Objective Cuts Remaining, ', str(philp.NumFeasibilityCuts()),
              ' Feasibility Cuts Remaining.')
        continue
    cS = philp.CandidateVector()

    philp.SolveSubProblems()
    totalCutsMade = totalCutsMade + 1
    philp.GenerateCuts()

    if exitFlag == -100:
        philp.ForceAcceptSolution()

    philp.UpdateSolutions()
    philp.UpdateTolerances()
    # philp.WriteProgress()
    #
    # print('Total cuts made: ' + str(totalCutsMade))
    # print('Total problems solved: ' + str(totalProblemsSolved))
    # print("=" * 100)

timeRuns=time.clock() - start
outTotalCuts = totalCutsMade
outTotalProbs = totalProblemsSolved

print(philp.ObjectiveValue(), '\n',
      philp.bestSolution.Mu(), '\n',
      philp.bestSolution.Lambda(), '\n',
      totalProblemsSolved, '\n',
      timeRuns-philp.timeruns1)
