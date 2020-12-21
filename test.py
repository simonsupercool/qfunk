import unittest
import numpy as np
from qfunk.functions import *






##############################################
# unit tests for random generation of things #
##############################################

class Testrand_rho(unittest.TestCase):

    # check if randomly produced density operator is valid density matrix
    def test_isdensity(self):
        # generate random state
        rho = rand_rho(10)

        # check if trace one
        self.assertTrue(np.isclose(np.trace(rho), 1.0), msg="matrix is not trace one")

        # check if hermitian
        self.assertTrue(np.allclose(rho, np.conj(np.transpose(rho))), msg="matrix is not hermitian")

        # check if positive semidefinite (matrix is known to be hermitian by this point so use Cholesky)
        try:
            # compute cholesky decomposition with small offset to allow for semidefiniteness
            np.linalg.cholesky(rho + np.eye(len(rho))*1e-9)
            # assert true outcome
            self.assertTrue(True)
        except LinAlgError:
            # assert False outcome
            self.assertFalse(False, msg="random state is not positive semidefinite")




##############################################
###### unit tests for partial operations #####
##############################################
class Testtrace_x(unittest.TestCase):

    # define a partial trace over product states
    def test_product(self):
        # dimension of random states
        dim_one = 10
        dim_two = 11

        # construct simple seperable states
        rand_rho_one = rand_rho(dim_one)
        rand_rho_two = rand_rho(dim_two)
        # compute tensor product
        state = np.kron(rand_rho_one, rand_rho_two)
        # now compute partial trace over both subsystems
        ptrace_one = trace_x(state, sys=[1], dim=[dim_one, dim_two])
        ptrace_two = trace_x(state, sys=[0], dim=[dim_one, dim_two])

        # perform checks on both subsystems
        self.assertTrue(np.allclose(ptrace_one, rand_rho_one), msg="seperable state B not correctly traced out")
        self.assertTrue(np.allclose(ptrace_two, rand_rho_two), msg="seperable state A not correctly traced out")

    # test a partial trace over maximally entangled state

    # test a partial trace over multipartite system

##############################################




# run tests
if __name__ == '__main__':
    unittest.main()