import sys
import os
sys.path.append("/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
import numpy as np
import cplex
from cplex.exceptions import CplexError
import scipy.io as sio
from scipy.sparse import find

class set(object):
    def __init__(self, mat_data):
        self.numStages = mat_data['numStages'][0][0].astype(int)
        self.numScenarios = mat_data['numScenarios'][0][0].astype(int)
        self.obs = mat_data['obs'][0].astype(float)

        # First stage
        self.first_obj = mat_data['first_obj'][0].astype(float)
        self.first_lb = mat_data['first_lb'][0].astype(float)
        self.first_ub = mat_data['first_ub'][0].astype(float)
        self.first_ub[np.isinf(self.first_ub)] = cplex.infinity
        self.first_rhs = mat_data['first_rhs'][0].astype(float)
        self.first_sense = ["E"] * self.first_rhs.size

        self.first_A = mat_data['first_A'].astype(float)
        A_rows = find(mat_data['first_A'])[0].tolist()
        A_cols = find(mat_data['first_A'])[1].tolist()
        A_vals = find(mat_data['first_A'])[2].tolist()
        self.first_coefficients = list(zip(A_rows, A_cols, A_vals))


        # Second stage

        self.second_obj = [mat_data['second_obj'][0][i][0].astype(float) for i in range(self.numScenarios)]
        self.second_lb = [mat_data['second_lb'][0][i][0].astype(float) for i in range(self.numScenarios)]
        self.second_ub = [mat_data['second_ub'][0][i][0].astype(float) for i in range(self.numScenarios)]
        for i in range(self.numScenarios):
            self.second_ub[i][np.isinf(self.second_ub[i])] = cplex.infinity
        self.second_rhs = [mat_data['second_rhs'][0][i][0].astype(float) for i in range(self.numScenarios)]
        self.second_sense = np.tile(["E"] * self.second_rhs[0].size, self.numScenarios).reshape(self.numScenarios, self.second_rhs[0].size)
        self.second_coefficients = [[] for i in range(self.numScenarios)]

        self.second_D = [mat_data['second_D'][0][i].astype(float) for i in range(self.numScenarios)]
        for j in range(self.numScenarios):
            D_rows = find(mat_data['second_D'][0][i])[0].tolist()
            D_cols = find(mat_data['second_D'][0][i])[1].tolist()
            D_vals = find(mat_data['second_D'][0][i])[2].tolist()
            self.second_coefficients[j] = list(zip(D_rows, D_cols, D_vals))
        self.second_B = [mat_data['second_B'][0][i].astype(float) for i in range(self.numScenarios)]



if __name__ == "__main__":
    # test
    try:
        mat_data = sio.loadmat(os.getcwd() + "/mat_data/apl1p15.mat")

        lp = set(mat_data)
        apl1p = cplex.Cplex()
        apl1p.set_problem_name("apl1p")
        apl1p.parameters.lpmethod.set(apl1p.parameters.lpmethod.values.auto)

        apl1p.objective.set_sense(apl1p.objective.sense.minimize)
        apl1p.variables.add(obj=lp.first_obj, lb=lp.first_lb, ub=lp.first_ub)
        apl1p.linear_constraints.add(senses=lp.first_sense, rhs=lp.first_rhs)
        apl1p.linear_constraints.set_coefficients(lp.first_coefficients)


        apl1p.solve()
        solution = apl1p.solution

        numvars = apl1p.variables.get_num()
        numlinconstr = apl1p.linear_constraints.get_num()

        x = np.array(solution.get_values(0, numvars - 1))
        print(x)

        apl1p_2nd = cplex.Cplex()
        apl1p_2nd.set_problem_name("apl1p_2nd")
        apl1p_2nd.parameters.lpmethod.set(apl1p_2nd.parameters.lpmethod.values.auto)

        apl1p_2nd.objective.set_sense(apl1p_2nd.objective.sense.minimize)
        apl1p_2nd.variables.add(obj=lp.second_obj[0], lb=lp.second_lb[0], ub=lp.second_ub[0])

        apl1p_2nd.linear_constraints.add(senses=lp.second_sense[0], rhs=lp.second_rhs[0] + x * lp.second_B[0].transpose())
        apl1p_2nd.linear_constraints.set_coefficients(lp.second_coefficients[0])
        apl1p_2nd.solve()
        solution = apl1p_2nd.solution

        numvars = apl1p_2nd.variables.get_num()
        numlinconstr = apl1p_2nd.linear_constraints.get_num()

        y = np.array(solution.get_values(0, numvars - 1))
        pi = solution.get_dual_values()
        print(y)
        print(pi)

    except CplexError as exc:
        print(exc)
