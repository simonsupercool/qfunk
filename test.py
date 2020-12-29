
import unittest

import numpy as np
from scipy.special import comb

from qfunk.utility import *
from qfunk.random import *
from qfunk.opensys import *
from qfunk.qoptic import *
from numpy import linalg as LA



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
###### unit tests for qoptics functions  #####
##############################################

class Test_symmetric_map(unittest.TestCase):
    # define test to check correct 
    def test_dimension(self):
        # define number of modes and photons
        m_num = 4
        p_num = 2

        # compute symmetric matrix
        S = symmetric_map(m_num, p_num)
        # computs shape of operator
        dims = np.shape(S)
        print(dims)
        # check output dimension is correct
        self.assertTrue(dims[0]==comb(m_num+p_num-1,p_num, exact=True))
        # check input direction is correct
        self.assertTrue(dims[1]==m_num**p_num)

    def test_isometry(self):
        # define number of modes and photons
        m_num = 4
        p_num = 2

        # compute symmetric matrix
        S = symmetric_map(m_num, p_num)
            
        # test for isometric transformation in direction of map (inverse is not one obviously)
        self.assertTrue(np.allclose(S @ np.transpose(S), np.eye(comb(m_num+p_num-1,p_num, exact=True))))


##############################################
###### unit tests for opensys functions  #####
##############################################

class Test_comb_func(unittest.TestCase):  
    
    #########
    #Check Link Product
    #########
    #check if Link Product of positive matrices is positive
    def test_positive(self):
        #generate random states, i.e., positive matrices
        c_1 = rand_rho(12)
        c_2 = rand_rho(12)
        comb_1 = Comb(c_1,[3,2,2],['A','B','C'])
        comb_2 = Comb(c_2,[2,3,2],['C','A','D'])
        #check if link product is positive
        self.assertTrue(min(np.real(LA.eigvals(comb_1.link(comb_2).mat))) >= 0, msg="Link product not positive")
        
    #check if Link Product of hermitian matrices is positive
    def test_hermitian(self):
        #generate random hermitian matrices
        c_1 = (np.random.rand(144) + 1j*np.random.rand(144)).reshape((12,12))
        c_1 = 0.5*(c_1 + np.transpose(np.conjugate(c_1)))
        c_2 = (np.random.rand(144) + 1j*np.random.rand(144)).reshape((12,12))
        c_2 = 0.5*(c_2 + np.transpose(np.conjugate(c_2)))
        comb_1 = Comb(c_1,[3,2,2],['A','B','C'])
        comb_2 = Comb(c_2,[2,3,2],['C','A','D'])
        c_3 = comb_1.link(comb_2).mat
        #check if link product is positive
        self.assertTrue(np.allclose(dagger(c_3),c_3), msg="Link product not Hermitian")
    
    #check if Link Product is commutative (up to reordering)
    def test_commute(self):
        #generate random complex matrices
        c_1 = (np.random.rand(144) + 1j*np.random.rand(144)).reshape((12,12))
        c_2 = (np.random.rand(144) + 1j*np.random.rand(144)).reshape((12,12))
        comb_1 = Comb(c_1,[3,2,2],['A','B','C'])
        comb_2 = Comb(c_2,[2,3,2],['C','A','D'])
        
        c_3 = comb_1.link(comb_2).mat
        c_4 = comb_2.link(comb_1).mat
        
        self.assertTrue(np.allclose(sys_permute(c_3, [1,0], [2,2]),c_4), msg="Link product not commutative")

    
    #########
    #Check relabel functions
    #########
    def test_relabel(self):
        #Generate random comb like object
        c_1 = (np.random.rand(144) + 1j*np.random.rand(144)).reshape((12,12))
        comb_1 = Comb(c_1,[3,2,2],['A','B','C'])
        #Check if replacing all labels works
        newlab = ['X','Y','Z']
        comb_1.relabel(newlab)
        self.assertTrue(comb_1.spaces == newlab, msg="Relabel function of comb class does not work")
        
        #Check if replacing some labels works
        
        #introduce new set of potential labels, choose random number of labels that should be exchanged
        Testlabels = np.array(['P','Q','R'])
        num_of_spaces = np.random.randint(len(newlab)-1)+1
        exchanged_spaces = np.random.choice(np.arange(len(newlab)), num_of_spaces, replace=False)
        
        #construct lists of old labels and new labels, as well as desired final list of labels
        original_spaces = np.array(comb_1.spaces)
        oldlab = original_spaces[exchanged_spaces]
        newlab = Testlabels[exchanged_spaces]
        original_spaces[exchanged_spaces] = Testlabels[exchanged_spaces]
        comb_1.relabelInd(oldlab,newlab)
        self.assertTrue((comb_1.spaces == original_spaces).all(), msg="RelabelInd function of comb class does not work")
    
# run tests
if __name__ == '__main__':
    unittest.main(verbosity=2)