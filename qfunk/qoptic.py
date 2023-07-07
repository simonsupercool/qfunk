
import numpy as np
from scipy.special import comb
from scipy.sparse import csc_matrix, lil_matrix
from itertools import product

import qfunk.utility as qut



def dirsum(arrayset, combination):
    """
    Generates direct product sum of arrayset according to given combination

    Parameters
    -----------
    arrayset: list of matrices to draw from e.g. [A,B,C,\dots]
    combination: list of integers that defines order of direct sum given arrayset e.g. [0,1,0,2,1]
    
    Requires
    -----------
    numpy as np
    
    Returns
    -----------
    Direct sum matrix given combination and array codebook
    """

    # initialise array with first element of combination`
    U = arrayset[combination[0]]
    # iterate over remaining elements
    for i in combination[1:]:
        # append next operator
        U = block_diag(U, arrayset[i])
    return U
    


def _str_base(num, base, length, numerals=100):
    """
    Outputs a list of number states in each mode, useful when needing consistent dimension labels in a Fock space.

    Parameters
    -----------
    num: number to be converted to base string
    base: base to work in
    length: number of characters to work to
    numerals: encoding scheme to use
    
    Requires
    -----------
    numpy as np
    
    Returns
    -----------
    Representation of input number given chosen base and word length in string format
    
    """
    # generate the list of numeral basis elements 
    numerals = [r for r in range(numerals)]

    if base < 2 or base > len(numerals):
        raise ValueError("str_base: base must be between 2 and %i" % len(numerals))

    if num == 0:
        return [0]*length

    result = []
    while num:
        result = [numerals[num % (base)]] + result
        num //= base

    if len(result) < length:
        result = [0]*(length - len(result)) + result

    return result






def fermi_to_boson(num, m, p):
     """
     Converts fermionic number states to bosonic number states. Map is surjective
     """

     # intialise list
     boson_state = []
     
     # iterate over fermionic digits
     for l in range(m):
          boson_state.append(sum(np.equal(num,l)))

     # return as immutable tuple
     return tuple(boson_state)



def symmetric_map(m_num, p_num):
     """
     Computes the permutation matrix to map the single Boson Hilbert space to that for p_num photons
     and m_num modes, eliminating degenerate degrees of freedom. Exponentially faster than matrix permanent method
     everyone else insists on using but at the cost of higher memory complexity. 
     Useful for computing the action of single photon unitaries on multiphoton input states as U_{p_num} = S U^{\otimes p_num} S^T. 

     Parameters
     -----------
     m_num: number of bosonic modes in system
     p_num: total number of bosons in system


     Requires
     -----------
     numpy as np

     Returns
     -----------
     An isometric operator S that takes operators defined on <p_num> subsystems each with <m_num> dof to the symmetric equivalent 

     """

     # compute size of output matrix
     row_num = comb(m_num + p_num-1, p_num, exact=True)
     col_num = m_num**p_num

     # initialise a dict to store index hits
     photonic_states = {}

     # preallocate symmetric transform matrix
     P = lil_matrix((row_num, col_num))

     # compute fermionic number states
     fermionic_number_states = [tuple(_str_base(n,m_num,p_num)) for n in range(m_num**p_num)]

     # iterate over ferminonic number states and compute corresponding bosonic state
     index_counter = 0
     for col_ind, fermi_state in enumerate(fermionic_number_states):
          
          # compute bosonic equivlaent
          boson_state = fermi_to_boson(fermi_state, m_num, p_num)

          # check if already had this number
          if boson_state in photonic_states:
               # get row index and update the symmetric map
               row_ind = photonic_states[boson_state]
               P[row_ind, col_ind] = 1

               continue

          # update our indexer and assign
          else:
               # update index dictionary
               photonic_states[boson_state] = index_counter
               
               # update symmetric map
               row_ind = photonic_states[boson_state]
               P[row_ind, col_ind] = 1

               # update counter
               index_counter += 1

     # reverse ordering to match fock state order
     P = P[::-1,::-1]


     # ensure normalisation property holds
     for k in range(row_num):
          P[k,:] /= np.sqrt(np.sum(P[k,:]))

     return P



def number_states(m_num,p_num):
    """
    Outputs a list of number states in each mode, useful when needing consistent dimension labels in a Fock space.

    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    
    Requires
    -----------
    numpy as np
    scipy.special.comb
    
    Returns
    -----------
    m_num choose p_num x m_num numpy array of bosonic number states
    
    """

    # compute size of output matrix
    row_num = comb(m_num + p_num-1, p_num)
    col_num = m_num**p_num

    # compute photon states as an m-ary number of p_num bits
    photon_state = np.asarray([list(_str_base(n,m_num,p_num)) for n in range(m_num**p_num)]).astype(int)

    # compute mode occupation number for each row
    fock = np.zeros((col_num, m_num), dtype=np.int32);
    # iterate over rows
    for i in range(np.shape(photon_state)[0]):
        # iterate over columns
        for j in range(m_num):
            fock[i,j] = np.sum(photon_state[i,:]==j)

    # compute unique Fock states
    uniques = np.fliplr(np.unique(fock, axis=0))
    return uniques


def fock_dim(m_num, p_num, full=False):
    """
    Computes dimension of symemtric Fock space given m_num modes and p_num bosons. 
    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    full: boolean specifying whether to return the dimenion of the direct sum space or just the largest
    
    Requires
    -----------
    scipy.special.comb

    
    Returns
    -----------
    Integer representing the dimension
    
    """
    if full:
        # base dimenion is vac state
        dim = 1
        for p in range(1,p_num+1):
            # compute dimenion of each photon subspace
            dim += comb(m_num+p-1,p, exact=True)
        return dim
    else:
        return comb(m_num+p_num-1,p_num, exact=True)




def lookup_gen(m_num, p_num, nstates=None):
    """
    Generate a lookup table for accessing indexes of number state basis - much more efficient that searching 
    the basis labels everytime.

    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    
    Requires
    -----------
    numpy as np

    
    Returns
    -----------
    dim x m_num array giving the index of a particular number state
    """

    # generate number states
    if nstates is None:
        nstates = number_states(m_num, p_num)

    # generate lookup table of correct dimension - TODO: turn this into a dictionary call - this is very memory inefficient
    dims = [p_num+1]*m_num
    lookup_table = {}
    # indexes of number state give corresponding index in table
    for i in range(len(nstates)):
        # assign index
        index = ''.join(map(str,nstates[i,:]))
        lookup_table[index] = i

    return lookup_table



def number_state_map(m_num, p_num, j,k, basis, nstates, lookup, sparse=False):
    """ 
    Computes jk basis element of an alegebra for optical transformations. 

    Ref: arXiv:1901.06178 

    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    
    Requires
    -----------
    numpy as np
    scipy.special.comb
    qfunk.qoptic.fock_dim
    

    Returns
    -----------
    m_num choose p_num x m_num numpy array of bosonic number states


    """

    # preallocate operator
    dim = fock_dim(m_num=m_num, p_num=p_num)
    operator = np.zeros((dim,dim), dtype=np.complex128)

    # iterate over all number states
    for i in range(np.shape(nstates)[0]):

        #  create copy of get target state
        curr_state = nstates[i,:].copy()

        # get occupation numbers for target modes
        create_boson_num = curr_state[j]
        ann_boson_num = curr_state[k]

        # no mapping occurs
        if (create_boson_num >= p_num or ann_boson_num <= 0) and k!=j:
            continue
        elif j==k:
            output_key = input_key = ''.join(map(str,nstates[i,:]))
        else:
            # else compute new number states
            curr_state[j] = curr_state[j] + 1
            curr_state[k] = curr_state[k] - 1
         
            # now compute basis index of output quantum state   
            input_key = ''.join(map(str,nstates[i,:]))
            output_key = ''.join(map(str,curr_state))

        
        # get basis indices from lookup table
        input_ind = lookup[input_key]
        output_ind = lookup[output_key]
        
        # using keys, construct map element
        input_state = basis[input_ind,:].reshape([1,-1])
        output_state = basis[output_ind,:].reshape([-1,1])

        # compute eigenvalue coefficients
        if j==k:
            coeff = np.sqrt(create_boson_num)**2
        else:
            coeff = np.sqrt(create_boson_num+1)*np.sqrt(ann_boson_num)

        operator += coeff*np.kron(output_state, input_state)

    if sparse:
        return csc_matrix(operator)
    else:
        return operator


def opt_subalgebra_gen(m_num, p_num):
    """
    Outputs a basis for the generating algebra of unitary transformations on m_num modes and p_num bosons.

    Ref: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.022301 

    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    
    Requires
    -----------
    numpy as np
    qfunk.qoptic.fock_dim
    qfunk.qoptic.number_states
    qfunk.qoptic.lookup_gen
    

    Returns
    -----------
    m_num choose p_num x m_num numpy array of bosonic number states


    
    """

    # catch trivial case
    if p_num==0:
        return np.asarray([[[1.0j]]])

    # compute dimension of isometric image
    dim = fock_dim(m_num, p_num, full=False)
    # generate number states for allocation 
    nstates = number_states(m_num=m_num, p_num=p_num)
    # generate lookup table for indexing
    lookup_table = lookup_gen(m_num=m_num, p_num=p_num)
    # preallocate single photon basis array
    basis = np.eye(dim)
    # initialise list to contain basis (allows for sparse representation)
    algebra_basis = np.zeros((m_num**2, dim, dim),dtype=np.complex128)

    # compute the basis for the multiphoton algebra
    cnt = 0
    for j in range(m_num): 
        for k in range(j+1):
            op_left = number_state_map(m_num=m_num, p_num=p_num, 
                             j=j,k=k, 
                             basis=basis, 
                             nstates=nstates, 
                             lookup=lookup_table)
            op_right = number_state_map(m_num=m_num, p_num=p_num, 
                             j=k,k=j, 
                             basis=basis, 
                             nstates=nstates, 
                             lookup=lookup_table)
            
            
            matrix = (op_left + op_right)*1j/2
            algebra_basis[cnt,:,:] = matrix
            cnt += 1

    for j in range(m_num):
        for k in range(j):

            op_left = number_state_map(m_num=m_num, p_num=p_num, 
                             j=j,k=k, 
                             basis=basis, 
                             nstates=nstates, 
                             lookup=lookup_table)

            op_right = number_state_map(m_num=m_num, p_num=p_num, 
                             j=k,k=j, 
                             basis=basis, 
                             nstates=nstates, 
                             lookup=lookup_table)
            
            matrix = (op_left - op_right)/2
            algebra_basis[cnt,:,:] = matrix
            cnt += 1

    return algebra_basis


def optical_state_gen(m_num, p_num, nstate, sparse=False):
    """
    Generate basis element of given number state such that it is consistent with qfunk labelling convention is. 

    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    nstate: list of integers of length m_num summing to p_num that define the desired number state

    Requires
    -----------
    numpy as np
    qfunk.qoptic.fock_dim
    qfunk.qoptic.lookup_gen
    

    Returns
    -----------
    density operator representing specified number state consistent qith qfunk labelling scheme

    """

    # compute lookup table
    lookup = lookup_gen(m_num=m_num, p_num=p_num)

    # compute index key
    key = ''.join(map(str,nstate))

    # generate basis
    basis = np.eye(fock_dim(m_num=m_num, p_num=p_num))
    # get basis element and return it
    element = basis[lookup[key],:].reshape([-1,1])
    rho = np.kron(element, qfunk.utility.dagger(element))

    if sparse:
        return csc_matrix(rho)
    else:
        return rho


def optical_projector(m_num, p_num, proj_modes, target, sparse=False):
    """
    Computes operator that maps bosons in given modes to a target mode - acts like a partial trace operation while preserving dimension,  
    Parameters
    -----------
    m_num: number of bosonic modes in system 
    p_num: total number of bosons in system
    nstate: list of integers of length m_num summing to p_num that define the desired number state

    Requires
    -----------
    numpy as np
    qfunk.qoptic.fock_dim
    qfunk.qoptic.lookup_gen
    

    Returns
    -----------
    density operator representing specified number state consistent qith qfunk labelling scheme

    """  

    # compute dimension of Fock spaces
    dim = fock_dim(m_num=m_num, p_num=p_num)

    # catch edge case
    if m_num==len(proj_modes):
        raise ValueError("Complete projection of state requested - is pointless, just generate the desired state")

    # generate number states for two spaces
    nstates = number_states(m_num=m_num, p_num=p_num)
    # generate lookup table
    lookup = lookup_gen(m_num=m_num, p_num=p_num)
    # generate basis elements
    basis = np.eye(dim, dtype=np.complex128)
    # initialise operator list (keeps sparsity)
    operators = []

    # iterate over all number states and compute corresponding map
    for i in range(dim):
        # compute input state
        input_state = nstates[i, :]

        # compute correct output state to map to 
        output_state = input_state.copy()
        # compute total number of photons in target modes
        mode_p = sum(output_state[proj_modes])
        output_state[proj_modes] = 0
        output_state[target] = mode_p

        # compute corresponding basis elements
        input_key = ''.join(map(str,input_state))
        output_key = ''.join(map(str,output_state))  

        input_element = basis[lookup[input_key],:].reshape([-1,1])
        output_element = basis[lookup[output_key],:].reshape([-1,1])

        # construct operator and add to list
        op = np.kron(output_element, qfunk.utility.dagger(input_element))

        # check if sparse matrix representation is requested
        if sparse:
            operators.append(csc_matrix(op))
        else:   
            operators.append(op)

    return operators


if __name__ == '__main__':

    algebra = opt_subalgebra_gen(2,5)


    # for i in range(len(algebra)):

    #     print(algebra[i,:].reshape([dim,dim]))