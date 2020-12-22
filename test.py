import unittest
import numpy as np
from qfunk.utility import *
from qfunk.random import *
from qfunk.opensys import *






##############################################
# unit tests for random generation of things #
##############################################

class Test_rand_rho(unittest.TestCase):

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



class Test_random_unitary(unittest.TestCase):
    # check if generated matrix is indeed unitary
    def test_isunitary(self):
        # choose dimension of matrix
        dim = 10

        # generate random unitary matrix
        U = random_unitary(dim)

        # compute left and right products
        left_op = U @ np.conjugate(np.transpose(U))
        right_op = np.conjugate(np.transpose(U)) @ U

        # confirm identity equivalency
        self.assertTrue(np.allclose(left_op, np.eye(dim)), msg="Product of UU^\dagger is not identity")
        self.assertTrue(np.allclose(right_op, np.eye(dim)), msg="Product of U^\dagger U is not identity")


##############################################
###### unit tests for utility operations #####
##############################################
class Test_trace_x(unittest.TestCase):

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
    def test_entangled(self):
        # dimension of subsystems
        dim = 10

        # generate me a maximally entangled density operator
        ent_state = ent_gen(dim, vec=False)

        # compute partial trace of both subsytems
        subsys_one = trace_x(ent_state, sys=[0], dim=[dim,dim])
        subsys_two = trace_x(ent_state, sys=[1], dim=[dim,dim])

        # assert both are equal to maximally mixed state
        self.assertTrue(np.allclose(subsys_one, np.eye(dim)/dim), msg="Partial trace of maximally entangled state is not maximally mixed")
        self.assertTrue(np.allclose(subsys_two, np.eye(dim)/dim), msg="Partial trace of maximally entangled state is not maximally mixed")

    # test a partial trace over multipartite system
    def test_ABC(self):
        # dimension of subsystems
        dim = 10

        # generate three random state
        A = rand_rho(dim)
        B = rand_rho(dim)
        C = rand_rho(dim)

        # compute product state
        ABC = np.kron(np.kron(A,B),C)

        # compute partial trace of all divisions
        AB = trace_x(ABC, sys=[2], dim=[dim,dim,dim])
        BC = trace_x(ABC, sys=[0], dim=[dim,dim,dim])
        AC = trace_x(ABC, sys=[1], dim=[dim,dim,dim])


        # assert all are equal to sub products
        self.assertTrue(np.allclose(AB, np.kron(A,B)), msg="Partial trace over C is not equal to AB")
        self.assertTrue(np.allclose(BC, np.kron(B,C)), msg="Partial trace over A is not equal to BC")
        self.assertTrue(np.allclose(AC, np.kron(A,C)), msg="Partial trace over B is not equal to AC")
       

class Test_dagger(unittest.TestCase):

    # define test of hermitian conjugate
    def test_hermitian_conjugate(self):
        # generate random state
        matrix = np.tril(np.ones((10,10))) + 1j*np.triu(np.ones((10,10)))
        # check if dagger is performing correct operation on hermitian matrix
        self.assertTrue(np.allclose(np.conj(np.transpose(matrix)), dagger(matrix)), msg="Hermitian conjugate not correctly computed")


class Test_eyelike(unittest.TestCase):
    # define test for identity generation of same dimension
    def test_generation(self):
        # generate identity matrix
        matrix_eye = np.eye(10)
        matrix_like = np.ones(10)

        # test function
        self.assertTrue(np.allclose(matrix_eye, eye_like(matrix_like)),msg="Generated matrix not similar to identity")



##############################################




# run tests
if __name__ == '__main__':
    unittest.main(verbosity=2)