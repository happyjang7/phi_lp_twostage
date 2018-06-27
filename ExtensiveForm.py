from __future__ import print_function
import sys, os
sys.path.append("/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
import numpy as np
import lp
import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import SimplexCallback
import scipy.io as sio
from scipy.sparse import csr_matrix, find, coo_matrix, hstack, vstack, block_diag

class MyCallback(SimplexCallback):
    def __call__(self):
        print("CB Iteration ", self.get_num_iterations(), " : ", end=' ')
        if self.is_primal_feasible():
            print("CB Objective = ", self.get_objective_value())
        else:
            print("CB Infeasibility measure = ",
                  self.get_primal_infeasibility())

def solve(lp):
    q = lp.obs/np.sum(lp.numScenarios)
    extensive_obj = lp.first_obj
    for i in range(lp.numScenarios):
        extensive_obj = np.append(extensive_obj,q[i]*lp.second_obj[i])

    extensive_lb = lp.first_lb
    for i in range(lp.numScenarios):
        extensive_lb = np.append(extensive_lb,lp.second_lb[i])

    extensive_ub = lp.first_ub
    for i in range(lp.numScenarios):
        extensive_ub = np.append(extensive_ub,lp.second_ub[i])

    extensive_rhs = lp.first_rhs
    for i in range(lp.numScenarios):
        extensive_rhs = np.append(extensive_rhs,lp.second_rhs[i])

    extensive_sense = lp.first_sense
    for i in range(lp.numScenarios):
        extensive_sense = np.append(extensive_sense,lp.second_sense[i])

    tmp = lp.first_A
    for i in range(lp.numScenarios):
        tmp = vstack([tmp,-lp.second_B[i]])

    tmp1 = csr_matrix(np.zeros((lp.first_rhs.size, lp.second_obj[0].size * lp.numScenarios)))
    tmp2 = lp.second_D[0]
    for i in range(1,lp.numScenarios):
        tmp2 = block_diag((tmp2,lp.second_D[i]))
    tmp3 = vstack((tmp1,tmp2))
    extensive_A = hstack([tmp, tmp3])
    A_rows = find(extensive_A)[0].tolist()
    A_cols = find(extensive_A)[1].tolist()
    A_vals = find(extensive_A)[2].tolist()
    extensive_A_coefficients = zip(A_rows, A_cols, A_vals)

    try:
        mdl = cplex.Cplex()
        mdl.set_problem_name("mdl")
        mdl.parameters.lpmethod.set(mdl.parameters.lpmethod.values.auto)

        mdl.objective.set_sense(mdl.objective.sense.minimize)
        mdl.variables.add(obj=extensive_obj, lb=extensive_lb, ub=extensive_ub)
        mdl.linear_constraints.add(senses=extensive_sense, rhs=extensive_rhs)
        mdl.linear_constraints.set_coefficients(extensive_A_coefficients)
        mdl.set_results_stream(None)
        mdl.solve()
        mdl.register_callback(MyCallback)
        solution = mdl.solution

        numvars = mdl.variables.get_num()
        x = np.array(solution.get_values(0, numvars - 1))
        exitFlag = solution.get_status()
        return (x, exitFlag)
    except CplexError as exc:
        print(exc)


if __name__ == "__main__":
    mat_data = sio.loadmat(os.getcwd() + "/mat_data/apl1p6.mat")
    lp = lp.set(mat_data)
    print(solve(lp))