import sys, os
sys.path.append("/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
import numpy as np
import lp
import cplex
import scipy.io as sio
from scipy.sparse import csr_matrix, find, hstack, vstack
import copy


def solve(lp):
    q = np.array([lp.obs / np.sum(lp.numScenarios)])

    expected_obj = np.append(lp.first_obj, np.sum(np.multiply(q.T, lp.second_obj), axis=0))
    expected_lb = np.append(lp.first_lb, np.sum(np.multiply(q.T, lp.second_lb), axis=0))
    expected_ub = np.append(lp.first_ub, np.sum(np.multiply(q.T, lp.second_ub), axis=0))
    expected_rhs = np.append(lp.first_rhs, np.sum(np.multiply(q.T, lp.second_rhs), axis=0))
    expected_sense = np.append(lp.first_sense, lp.second_sense[0])

    tmp = q[0][0] * (-lp.second_B[0])
    tmp1 = q[0][0] * lp.second_D[0]
    for i in range(1, lp.numScenarios):
        tmp = tmp + q[0][i] * (-lp.second_B[i])
        tmp1 = tmp1 + q[0][i] * lp.second_D[i]

    tmp2 = vstack([lp.first_A, tmp])
    tmp3 = csr_matrix(np.zeros((lp.first_rhs.size, lp.second_obj[0].size)))
    tmp4 = vstack([tmp3, tmp1])

    expected_A = hstack([tmp2, tmp4])
    A_rows = find(expected_A)[0].tolist()
    A_cols = find(expected_A)[1].tolist()
    A_vals = find(expected_A)[2].tolist()
    expected_A_coefficients = zip(A_rows, A_cols, A_vals)


    mdl = cplex.Cplex()
    mdl.set_problem_name("mdl")
    mdl.variables.add(obj=expected_obj, lb=expected_lb, ub=expected_ub)
    mdl.linear_constraints.add(senses=expected_sense, rhs=expected_rhs)
    mdl.linear_constraints.set_coefficients(expected_A_coefficients)
    mdl.set_results_stream(None)
    mdl.solve()
    solution = mdl.solution
    exitFlag = solution.get_status()

    slack = solution.get_linear_slacks()
    pi = solution.get_dual_values()
    x = solution.get_values()
    dj = solution.get_reduced_costs()
    return (np.array(x), exitFlag, np.array(slack), np.array(pi), np.array(dj))


if __name__ == "__main__":
    mat_data = sio.loadmat(os.getcwd() + "/mat_data/NI48.mat")
    lp = lp.set(mat_data)
    x, exitFlag, slack, pi, dj=solve(lp)

    dj_l = copy.deepcopy(dj)
    dj_l[dj_l < 0] = 0
    dj_u = copy.deepcopy(dj)
    dj_u[dj_u > 0] = 0

    print(list(dj_l))
    print(list(dj_u))

