
import numpy as np
from scipy.special import comb




def _str_base(num, base, length, numerals = '0123456789'):
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
    representation of input number given chosen base and word length in string format
    
    """

    if base < 2 or base > len(numerals):
        raise ValueError("str_base: base must be between 2 and %i" % len(numerals))

    if num == 0:
        return '0'*length

    if num < 0:
        sign = '-'
        num = -num
    else:
        sign = ''

    result = ''
    while num:
        result = numerals[num % (base)] + result
        num //= base


    out = sign + result

    if len(out) < length:
        out = '0'*(length - len(out)) + out
    return out



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
    An isometric operator S that takes operators defined for F_{m_num,1} to F_{m_num, p_num}. 
    
    """

    # compute size of output matrix
    row_num = comb(m_num + p_num-1, p_num)
    col_num = m_num**p_num

    # compute photon states as an m-ary number of p_num bits
    photon_state = np.asarray([list(_str_base(n,m_num,p_num)) for n in range(m_num**p_num)]).astype(int)

    # compute mode occupation number for each row
    fock = np.zeros((col_num, m_num), dtype=np.int32)
    # iterate over rows
    for i in range(np.shape(photon_state)[0]):
        # iterate over columns
        for j in range(m_num):
            fock[i,j] = np.sum(photon_state[i,:]==j)

    # compute unique Fock states
    uniques = np.fliplr(np.unique(fock, axis=0))
    ldim = np.shape(uniques)[0]

    # preallocate symmetric transform matrix
    P = np.zeros((ldim, col_num))

    # iterate over symmetric dimension
    for k in range(ldim):
        for m in range(col_num):
            if (uniques[k,:] == fock[m,:]).all():
                P[k,m] = 1
        
        # ensure normalisation property holds
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
