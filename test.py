
import unittest

import numpy as np
from scipy.special import comb
from numpy import linalg as LA

import qfunk.utility as qut
import qfunk.random as qr
import qfunk.opensys as qos
import qfunk.qoptic as qop




##############################################
# unit tests for random generation of things #
##############################################

class Test_rand_rho(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # define test dimension 
        self.dim = 10

        # call init of parent test class
        super(Test_rand_rho, self).__init__(*args, **kwargs)

    # check if randomly produced density operator is valid density matrix
    def test_isdensity(self):
        # generate random state
        rho = qr.rand_rho(self.dim)

        # check if trace one
        self.assertTrue(np.isclose(np.trace(rho), 1.0), msg="matrix is not trace one")

        # check if hermitian
        self.assertTrue(np.allclose(rho, np.conj(np.transpose(rho))), msg="matrix is not hermitian")

        # check if positive semidefinite (matrix is known to be hermitian by this point so use Cholesky)
        try:
            # compute cholesky decomposition with small offset to allow for semidefiniteness
            np.linalg.cholesky(rho + np.eye(self.dim)*1e-9)
            # assert true outcome
            self.assertTrue(True)
        except LinAlgError:
            # assert False outcome
            self.assertFalse(False, msg="random state is not positive semidefinite")

class Test_random_unitary(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        # define test dimension 
        self.dim = 10

        # call init of parent test class
        super(Test_random_unitary, self).__init__(*args, **kwargs)

    # check if generated matrix is indeed unitary
    def test_isunitary(self):

        # generate random unitary matrix
        U = qr.random_unitary(self.dim)

        # compute left and right products
        left_op = U @ np.conjugate(np.transpose(U))
        right_op = np.conjugate(np.transpose(U)) @ U

        # confirm identity equivalency
        self.assertTrue(np.allclose(left_op, np.eye(self.dim)), msg="Product of UU^\dagger is not identity")
        self.assertTrue(np.allclose(right_op, np.eye(self.dim)), msg="Product of U^\dagger U is not identity")


class Test_MUB_gen(unittest.TestCase):

    # setup test case
    def __init__(self, *args, **kwargs):
        # define test dimension (must be prime number)
        self.dim = 7
        # generate MUB
        self.MUB = qr.MUB_gen(self.dim)

        # call init of parent test class
        super(Test_MUB_gen, self).__init__(*args, **kwargs)

    # check if generated MUB produces positive states
    def test_isdensity(self):

        # iterate over each density operator and check for PSD
        for i in range(self.dim+1):
            for j in range(self.dim):

                # extract element
                rho = self.MUB[i,j,:,:]
                # check if trace one
                self.assertTrue(np.isclose(np.trace(rho), 1.0), msg="MUB is not trace one")
                # check if hermitian
                self.assertTrue(np.allclose(rho, np.conj(np.transpose(rho))), msg="MUB is not hermitian")

                # do most expensive check last
                try:
                    # compute cholesky decomposition with small offset to allow for semidefiniteness
                    np.linalg.cholesky(rho + np.eye(len(rho))*1e-9)
                    # assert true outcome
                    self.assertTrue(True)
                except LinAlgError:
                    # assert False outcome
                    self.assertFalse(False, msg="MUB state is not positive semidefinite")


    # check if generated MUB produces pure states
    def test_ispure(self):

        # iterate over each density operator and check for purity
        for i in range(self.dim+1):
            for j in range(self.dim):
                rho = self.MUB[i,j,:,:]

                self.assertTrue(np.isclose(np.trace(rho @ rho),1.0), msg="MUB state is not pure")

    # check if all basis collection satisfies MUB property
    def test_ismub(self):
        # iterate over each density operator and check for purity
        for i in range(self.dim+1):
            for j in range(self.dim):
                for k in range(self.dim+1):
                    for l in range(self.dim):
                        rho1 = self.MUB[i,j,:,:]
                        rho2 = self.MUB[k,l,:,:]

                        # check for relation
                        if i==k:
                            if j==l:
                                # already checking for purity
                                continue
                            # check for orthogonality
                            else:
                                self.assertTrue(np.isclose(np.trace(rho1 @ rho2) , 0.0), msg="basis elements are not orthogonal")

                        else:
                            # check for unbiasedness
                            self.assertTrue(np.isclose(np.trace(rho1 @ rho2) , 1/self.dim), msg="bases are not unbiased")
                        

##############################################
###### unit tests for utility operations #####
##############################################
class Test_trace_x(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        # define test dimension 
        self.dim_one = 10
        self.dim_two = 11

        # call init of parent test class
        super(Test_trace_x, self).__init__(*args, **kwargs)

    # define a partial trace over product states
    def test_product(self):
        # dimension of random state

        # construct simple seperable states
        rand_rho_one = qr.rand_rho(self.dim_one)
        rand_rho_two = qr.rand_rho(self.dim_two)
        # compute tensor product
        state = np.kron(rand_rho_one, rand_rho_two)
        # now compute partial trace over both subsystems
        ptrace_one = qut.trace_x(state, sys=[1], dim=[self.dim_one, self.dim_two])
        ptrace_two = qut.trace_x(state, sys=[0], dim=[self.dim_one, self.dim_two])

        # perform checks on both subsystems
        self.assertTrue(np.allclose(ptrace_one, rand_rho_one), msg="seperable state B not correctly traced out")
        self.assertTrue(np.allclose(ptrace_two, rand_rho_two), msg="seperable state A not correctly traced out")

    # test a partial trace over maximally entangled state
    def test_entangled(self):

        # generate me a maximally entangled density operator
        ent_state = qr.ent_gen(self.dim_one, vec=False)

        # compute partial trace of both subsytems
        subsys_one = qut.trace_x(ent_state, sys=[0], dim=[self.dim_one,self.dim_one])
        subsys_two = qut.trace_x(ent_state, sys=[1], dim=[self.dim_one,self.dim_one])

        # assert both are equal to maximally mixed state
        self.assertTrue(np.allclose(subsys_one, np.eye(self.dim_one)/self.dim_one), msg="Partial trace of maximally entangled state is not maximally mixed")
        self.assertTrue(np.allclose(subsys_two, np.eye(self.dim_one)/self.dim_one), msg="Partial trace of maximally entangled state is not maximally mixed")

    # test a partial trace over multipartite system
    def test_ABC(self):

        # generate three random state
        A = qr.rand_rho(self.dim_one)
        B = qr.rand_rho(self.dim_one)
        C = qr.rand_rho(self.dim_one)

        # compute product state
        ABC = np.kron(np.kron(A,B),C)

        # compute partial trace of all divisions
        AB = qut.trace_x(ABC, sys=[2], dim=[self.dim_one]*3)
        BC = qut.trace_x(ABC, sys=[0], dim=[self.dim_one]*3)
        AC = qut.trace_x(ABC, sys=[1], dim=[self.dim_one]*3)


        # assert all are equal to sub products
        self.assertTrue(np.allclose(AB, np.kron(A,B)), msg="Partial trace over C is not equal to AB")
        self.assertTrue(np.allclose(BC, np.kron(B,C)), msg="Partial trace over A is not equal to BC")
        self.assertTrue(np.allclose(AC, np.kron(A,C)), msg="Partial trace over B is not equal to AC")
       

class Test_dagger(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        # define test dimension 
        self.dim = 10

        # call init of parent test class
        super(Test_dagger, self).__init__(*args, **kwargs)

    # define test of hermitian conjugate
    def test_hermitian_conjugate(self):
        # generate random state
        matrix = np.tril(np.ones((self.dim,self.dim))) + 1j*np.triu(np.ones((self.dim,self.dim)))
        # check if dagger is performing correct operation on hermitian matrix
        self.assertTrue(np.allclose(np.conj(np.transpose(matrix)), qut.dagger(matrix)), msg="Hermitian conjugate not correctly computed")


class Test_eyelike(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # define test dimension 
        self.dim = 10

        # call init of parent test class
        super(Test_eyelike, self).__init__(*args, **kwargs)

    # define test for identity generation of same dimension
    def test_generation(self):
        # generate identity matrix
        matrix_eye = np.eye(self.dim)
        matrix_like = np.ones(self.dim)

        # test function
        self.assertTrue(np.allclose(matrix_eye, qut.eye_like(matrix_like)),msg="Generated matrix not similar to identity")



##############################################
###### unit tests for qoptics functions  #####
##############################################

class Test_symmetric_map(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # define number of modes and photons 
        self.m_num = 4
        self.p_num = 2

        # compute system dimension
        self.dim = comb(self.m_num+self.p_num-1,self.p_num, exact=True)

        # compute symmetric operator
        self.S = qop.symmetric_map(self.m_num, self.p_num)

        # call init of parent test class
        super(Test_symmetric_map, self).__init__(*args, **kwargs)

    # define test to check correct 
    def test_dimension(self):
        # computs shape of operator
        dims = np.shape(self.S)
        # check output dimension is correct
        self.assertTrue(dims[0]==self.dim)
        # check input direction is correct
        self.assertTrue(dims[1]==self.m_num**self.p_num)

    def test_isometry(self):
        # test for isometric transformation in direction of map (inverse is not one obviously)
        self.assertTrue(np.allclose(self.S @ np.transpose(self.S), np.eye(self.dim)))


##############################################
###### unit tests for opensys functions  #####
##############################################

class Test_Comb(unittest.TestCase):  
    
    #########
    #Check Link Product
    #########
    def __init__(self, *args, **kwargs):
        # define test dimension 
        self.dim = 12

        # define subsystem dimensions
        self.order1 = [3,2,2]
        self.order2 = [2,3,2]

        # call init of parent test class
        super(Test_Comb, self).__init__(*args, **kwargs)

    #check if Link Product of positive matrices is positive
    def test_positive(self):

        #generate random states, i.e., positive matrices
        c_1 = qr.rand_rho(self.dim)
        c_2 = qr.rand_rho(self.dim)
        comb_1 = qos.Comb(c_1, self.order1, ['A','B','C'])
        comb_2 = qos.Comb(c_2, self.order2, ['C','A','D'])

        #check if link product is positive
        self.assertTrue(min(np.real(LA.eigvals(comb_1.link(comb_2).mat))) >= 0, msg="Link product not positive")
        
    #check if Link Product of hermitian matrices is positive
    def test_hermitian(self):

        #generate random hermitian matrices
        c_1 = (np.random.rand(self.dim**2) + 1j*np.random.rand(self.dim**2)).reshape((self.dim,self.dim))
        c_1 = 0.5*(c_1 + np.transpose(np.conjugate(c_1)))
        c_2 = (np.random.rand(self.dim**2) + 1j*np.random.rand(self.dim**2)).reshape((self.dim,self.dim))
        c_2 = 0.5*(c_2 + np.transpose(np.conjugate(c_2)))
        comb_1 = qos.Comb(c_1, self.order1, ['A','B','C'])
        comb_2 = qos.Comb(c_2, self.order2, ['C','A','D'])
        c_3 = comb_1.link(comb_2).mat
        #check if link product is positive
        self.assertTrue(np.allclose(qut.dagger(c_3), c_3), msg="Link product not Hermitian")
    
    #check if Link Product is commutative (up to reordering)
    def test_commute(self):

        #generate random complex matrices
        c_1 = (np.random.rand(self.dim**2) + 1j*np.random.rand(self.dim**2)).reshape((self.dim, self.dim))
        c_2 = (np.random.rand(self.dim**2) + 1j*np.random.rand(self.dim**2)).reshape((self.dim, self.dim))
        comb_1 = qos.Comb(c_1, self.order1, ['A','B','C'])
        comb_2 = qos.Comb(c_2, self.order2, ['C','A','D'])
        
        c_3 = comb_1.link(comb_2).mat
        c_4 = comb_2.link(comb_1).mat
        
        # choose subsystems
        self.assertTrue(np.allclose(qut.sys_permute(c_3, [1,0], [2,2]),c_4), msg="Link product not commutative")

    
    #########
    #Check relabel functions
    #########
    def test_relabel(self):
        # define dimension of system
        dim = 12
        # define comb ordering
        order = [3,2,2]

        #Generate random comb like object
        c_1 = (np.random.rand(dim**2) + 1j*np.random.rand(dim**2)).reshape((dim,dim))
        comb_1 = qos.Comb(c_1, order,['A','B','C'])
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
        
    #########
    #Check is_causally_ordered functions
    #########
    def test_is_causal(self):
        #unnormalized maximally entangled state
        MaxEnt = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
        
        #Generate combs that end and start on different types of spaces (input/output)
        
        #four Hilbert spaces, begins on output, ends on input
        Mat1 = np.kron(MaxEnt,MaxEnt)
        spaces1 = ['A','B','C','D']
        dims1 = [2,2,2,2]
        comb_1 = qos.Comb(mat=Mat1,dims=dims1,spaces=spaces1)
        
        #three Hilbert spaces, begins on output, ends on output
        Mat2 = np.kron(MaxEnt,np.eye(3))
        spaces2 = ['A','B','C']
        dims2 = [2,2,3]
        comb_2 = qos.Comb(mat=Mat2,dims=dims2,spaces=spaces2)
        
        #three Hilbert spaces, begins on input, ends on input
        Mat3 = np.kron(np.eye(3)/3.,MaxEnt)
        spaces3 = ['A','B','C']
        dims3 = [3,2,2]
        comb_3 = qos.Comb(mat=Mat3,dims=dims3,spaces=spaces3)
        
        #three Hilbert spaces, begins on input, ends on input, different structure than comb_3
        Mat4 = qut.sys_permute(np.kron(np.eye(3),MaxEnt/2),[1,0,2],[3,2,2])
        spaces4 = ['A','B','C']
        dims4 = [2,3,2]
        comb_4 = qos.Comb(mat=Mat4,dims=dims4,spaces=spaces4)
    
        #Check if causality_check functions properly
        self.assertTrue(comb_1.is_causally_ordered(spaces1), msg="Causality check function of comb class does not work")
        self.assertTrue(comb_2.is_causally_ordered(spaces2), msg="Causality check function of comb class does not work")
        self.assertTrue(comb_3.is_causally_ordered(spaces3), msg="Causality check function of comb class does not work")
        self.assertTrue(comb_4.is_causally_ordered(spaces4), msg="Causality check function of comb class does not work")
        
# run tests
if __name__ == '__main__':
    unittest.main(verbosity=2)