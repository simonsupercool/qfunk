import numpy as np
import sympy
from numpy.linalg import det, solve
# SciPy instalation: in the cmd: 'py -m pip install scipy'
from scipy.linalg import expm
import qfunk.utility as qut
import qfunk.qoptic as qop
import qfunk.generator as qog
import matplotlib.pyplot as plt


# Adjoint representation for an U matrix
def adjoint_U(iH_U,U):

    return U.dot(iH_U.dot(np.transpose(np.conj(U))))




# The functions e_jk y f_jk allow to obtain the matrix basis of u(m)
def e_jk(j,k,base):

    j_array=np.array([base[j]])

    k_array=np.array([base[k]])

    ejk=0.5j*(np.transpose(j_array).dot(np.conj(k_array))+np.transpose(k_array).dot(np.conj(j_array)))

    return ejk


def f_jk(j,k,base):

    j_array=np.array([base[j]])

    k_array=np.array([base[k]])

    fjk=0.5*(np.transpose(j_array).dot(np.conj(k_array))-np.transpose(k_array).dot(np.conj(j_array)))

    return fjk


# We transform from the u(m) matrix basis to u(M)'s
def d_phi(base_matrix_m,photons,base_input):

    m=len(base_matrix_m)
    num_photons=int(np.sum(photons))

    if base_input==True:

        try:

            # We load the vector basis
            vec_base_file=open(f"m_{m}_n_{num_photons}_vec_base.txt","r")

            vec_base=np.loadtxt(vec_base_file,delimiter=",",dtype=complex) 

            vec_base_file.close()

        except FileNotFoundError:
                               
            print("\nThe required vector basis file does not exist.\n")
            print("\nIt will be freshly generated instead.\n")

            # We load the combinations with the same amount of photons in order to create the vector basis
            vec_base=photon_combs_generator(m,photons)

            vec_base_file=open(f"m_{m}_n_{num_photons}_vec_base.txt","w")

            np.savetxt(vec_base_file,vec_base,fmt="(%e)",delimiter=",")

            vec_base_file.close()

    else:

        # We load the combinations with the same amount of photons in order to create the vector basis
        vec_base=photon_combs_generator(m,photons)
    
    # It is required to introduce photons_aux for 'photons_aux' and 'photons' not to "update" together
    global photons_aux
    global mult

    # Dimensions of the resulting matrix U:
    M = qop.fock_dim(m, num_photons)
    # This value can be otained too with the measurement of vec_base's length

    # base_matrix_M initialization
    base_matrix_M=np.zeros((M,M),dtype=complex)

    base_matrix_M=u_m_to_u_M(m,M,vec_base,base_matrix_m)

    return base_matrix_M

# We define the operators a (annihilation) and a_dagger (creator):
def a(num_vec,array,mult):

    n=array[num_vec]

    mult*=np.sqrt(n)
    array[num_vec]=n-1

    return array, mult


def a_dagger(num_vec,array,mult):

    n=array[num_vec]

    mult*=np.sqrt(n+1)
    array[num_vec]=n+1

    return array, mult

# Specific process of subspace shift. Very similar to iH_U_operator's obtention (see '_2_3rd_evolution_method.py')
def u_m_to_u_M(m,M,vec_base,base_matrix_m):

    # base_matrix_M initialization
    base_matrix_M=np.zeros((M,M),dtype=complex)

    vec_base_canon=np.identity(M,dtype=complex)

    for p in range(M):

        p_array=np.array(vec_base[p])

        p_array_M=np.array(vec_base_canon[p])

        for q in range(M):

            q_array=np.array(vec_base[q])

            for j in range(m):

                for l in range(m):

                    # Array subject to the operators
                    q_array_aux=np.array(vec_base[q])

                    # Multiplier
                    mult=base_matrix_m[j,l]

                    # These two functions update q_array_aux and mult
                    q_array_aux,mult=a(l,q_array_aux,mult)

                    q_array_aux,mult=a_dagger(j,q_array_aux,mult)

                    for k in range(M):

                        if (vec_base[k]==q_array_aux).all():

                            index=k

                            break

                    q_array_M=np.array(vec_base_canon[index])

                    base_matrix_M[p,q]+=p_array_M.dot(q_array_M)*mult

    return base_matrix_M



def matrix_u_basis_generator(m,M,photons,base_input):

        # We initialise the basis for each space
        base_U_m=np.identity(m,dtype=complex)

        base_U_M=np.identity(M,dtype=complex)

        base_u_m=np.zeros((m*m,m,m),dtype=complex)

        base_u_M=np.zeros((M*M,M,M),dtype=complex)

        # Here we will storage correlations with e_jk and f_jk, for a better organisation
        base_u_m_e=np.zeros((m*m,m,m),dtype=complex)

        base_u_m_f=np.zeros((m*m,m,m),dtype=complex)

        cont=0

        for j in range(m):

            for k in range(m):

                base_u_m_e[m*j+k]=e_jk(j,k,base_U_m)

                if k<=j:

                    base_u_m[cont]=e_jk(j,k,base_U_m)
                    base_u_M[cont]=d_phi(base_u_m[cont],photons,base_input)

                    cont+=1

        # The separator's functions indicate the switch from e_jk to f_jk, 
        # after the m*m combinations have been already computed in the former
        separator_e_f=cont

        for j in range(m):

            for k in range(m):

                base_u_m_f[m*j+k]=f_jk(j,k,base_U_m)

                if k<j:

                    base_u_m[cont]=f_jk(j,k,base_U_m)

                    base_u_M[cont]=d_phi(base_u_m[cont],photons,base_input)

                    cont+=1

        return base_u_m, base_u_M, base_u_m_e, base_u_m_f, separator_e_f, base_U_m, base_U_M

# A selection of linear independent equations is obtained
def eq_sys_finder(base_u_m,base_u_M):

    m=len(base_u_m[0])
    M=len(base_u_M[0])

    # Equation system initialization
    eq_sys=np.zeros((M*M,m*m),dtype=complex)

    # Storage of all the present equations in the system
    for j in range(m*m):

        for l in range(M):

            for o in range(M):

                eq_sys[M*l+o,j]=base_u_M[j,l,o]

    # Array wich storages m*m equations of eq_sys, for which we will attempt to solve the system
    # We will use np.append() in this and the following array for adding new terms
    eq_sys_choice=np.zeros((1,m*m),dtype=complex)

    # Array which storages the indexes of the chosen equations
    index_choice=np.zeros(1,dtype=int)

    cont=0

    end=False

    # This loop searches for m*m equations of the list eq_sys for which a matrix with
    # a no-null determinant is made. That is, they are linear independent
    for l in range(M):

        for o in range(M):

            if cont>0:
                
                # With this functions, we conserve the linear independent rows
                aux,inds=sympy.Matrix(eq_sys_choice).T.rref()
                
                # Applying inds to our two arrays, the algorithm is still ongoing until...
                eq_sys_choice=np.array(eq_sys_choice[np.array(inds)])
                index_choice=np.array(index_choice[np.array(inds)])

            # By obtaining a m*m x m*m equation system, with a no-null determinant, we have
            # computed the required system
            if len(eq_sys_choice[0])==len(eq_sys_choice[:,0]) and det(eq_sys_choice)!=0:

                end=True

                break

            # This simple condition saves multiple steps of null vectors being eliminated
            elif (eq_sys[M*l+o]!=0).any():

                if cont==0:

                    eq_sys_choice[cont]=eq_sys[M*l+o]
                    index_choice[cont]=M*l+o

                else:

                    # We add the new arrays
                    eq_sys_choice=np.append(eq_sys_choice,np.array([eq_sys[M*l+o]]),axis=0)
                    index_choice=np.append(index_choice,np.array([M*l+o]),axis=0)

                cont+=1

        if end==True:

            break

    return eq_sys, eq_sys_choice, index_choice


def verification(U,base_u_m,base_u_m_e,base_u_m_f,sep,base_u_M,eq_sys,eq_sys_choice,index_choice):

    m=len(base_u_m[0])
    M=len(base_u_M[0])

    # Solution arrays initialization
    sol=np.zeros((m*m,m*m),dtype=complex)
    sol_e=np.zeros((m*m,m*m),dtype=complex)
    sol_f=np.zeros((m*m,m*m),dtype=complex)

    # Saving both basis of the u(m) and u(M) subspaces

    for j in range(m*m):

        # We compute the adjoint for each matrix in the basis of u(M)
        adj_U_b_j=adjoint_U(base_u_M[j],U)
        adj_U_b_j_reshape=np.reshape(adj_U_b_j,M*M)

        # We choose the adj_U_b_j values of the indexes corresponding to the used equations
        adj_U_b_j_choice=np.array(adj_U_b_j_reshape[np.array(index_choice)])
        
        sol[j]=solve(eq_sys_choice,adj_U_b_j_choice)

        # # Check for its validity for all possible equations?
        # for l in range(M*M):

        #     suma=0

        #     for o in range(m*m):

        #         suma+=eq_sys[l,o]*sol[j,o]

        #     if np.round(suma,5)!=np.round(adj_U_b_j_reshape[l],5):

        #         op=np.array([None])
        #         check=False

        #         # We return it three times for keeping consistency with the main algorithm 3
        #         return op,op,op,check



    # If the algorithm reaches this line, the solution exists. It is computed, giving a general solution of all
    # equations, and a separated version only applied to the e_jk and f_jk respectively, useful in the reconstruction of S
    check=True
    for j in range(m):

        for k in range(m):

            if m*j+k<sep:

                for l in range(m*m):

                    if (base_u_m_e[l]==base_u_m[m*j+k]).all():

                        sol_e[l]=sol[m*j+k]

            else:

                for l in range(m*m):

                    if (base_u_m_f[l]==base_u_m[m*j+k]).all():

                        sol_f[l]=sol[m*j+k]

                    if (base_u_m_f[l]==-base_u_m[m*j+k]).all():

                        sol_f[l]=-sol[m*j+k]

    return sol,sol_e,sol_f,check

# Adjoint representation for a S matrix
def adjoint_S(index,base_u_m,sol):

    m=len(base_u_m[0])

    suma=np.zeros((m,m),dtype=complex)

    for j in range(m*m):

        suma+=sol[index,j]*base_u_m[j]

    return suma


# Main function of S rebuilding
def S_output(base_u_m,base_U_m,sol_e,sol_f):

    m=len(base_u_m[0])

    S=np.zeros((m,m),dtype=complex)

    # First of all, we obtain a no-null value of S for using it as a base for the rest computations
    for l in range(m):

        end=False

        for j in range(m):

            l_array=np.array([base_U_m[l]])

            absS=-1j*np.conj(l_array).dot(adjoint_S(m*j+j,base_u_m,sol_e).dot(np.transpose(l_array)))

            # 8 decimal accuracy, it can be modified
            if np.round(absS,5)==0:

                S[l,j]=0

            else:

                # We ignore the offset (for now...)

                l0=l
                j0=j

                end=True

                break

        if end:

            break

    # Later, we compute the total matrix. l0 y j0 serve as a support
    for l in range(m):

        for j in range(m):

            l0_array=np.array([base_U_m[l0]])

            l_array=np.array([base_U_m[l]])

            j_array=np.array([base_U_m[j]])

            # Storage of the sum in S
            S+=(np.conj(l_array).dot(adjoint_S(m*j+j0,base_u_m,sol_f).dot(np.transpose(l0_array)))-1j*np.conj(l_array).dot(adjoint_S(m*j+j0,base_u_m,sol_e).dot(np.transpose(l0_array))))/ \
             np.sqrt(-1j*np.conj(l0_array).dot(adjoint_S(m*j0+j0,base_u_m,sol_e).dot(np.transpose(l0_array))))*(np.transpose(l_array).dot(np.conj(j_array)))

    return S



# Main function to inherit in other algorithms
def photon_combs_generator(m,photons):
    
    global photons_aux
    global vec_base
    global check

    check=0

    num_photons=int(np.sum(photons))

    counter=np.array(photons[:],dtype=int)

    counter_sum=np.zeros(num_photons,dtype=int)

    # The last two terms are required because of the function's recursive character
    photon_combs_generator_loop(photons,num_photons,m,counter_sum,0)

    return vec_base


# Loop whose amount of callings depend on the number of photons in each mode
def photon_combs_generator_loop(photons,num_photons,m,sum_,k):

    global photons_aux
    global vec_base
    global check

    counter=np.array(photons[:],dtype=int)

    for sum_[k] in range(m):

        if k<num_photons-1:

            photon_combs_generator_loop(photons,num_photons,m,sum_,k+1)

        else:

            photons_aux=np.zeros(m,dtype=complex)

            cont=0 # IMPORTANT, we want to explore sum_[] in order

            for p in range(m):

                for q in range(counter[p]):

                    photons_aux[sum_[cont]]+=1

                    cont+=1

            if check!=0:

                vec_base=photon_comb_basis(photons_aux,vec_base)

            else:

                vec_base=np.array([np.real(photons_aux)])

                check=1
        

# Required vector basis creator
def photon_comb_basis(array,vec_base):

    num_lines=len(vec_base[:,0]) # Reads all lines

    check=0

    for i in range(num_lines):

        lect=vec_base[i]
        
        if (array==lect).all(): # Reads a line

            check=1

            break

    if check==0:

        vec_base=np.insert(vec_base,len(vec_base),np.real(array),axis=0)

    return vec_base


def SfromU(file_input=True,U=False,file_output=True,filename=False,base_input=False,m=False,n=False,perm=False,acc_d=3,txt=False):
    '''
    Loads .txt files containing an evolution matrix U. Should it be buildable via linear optics elements, its scattering matrix of origin S will be rebuilt. Modes can be permutted for different ways of placing the instruments.
    Information is displayed on-screen.
    '''
    # M value (comb_evol(n,m)=comb(m+n-1,n))
    
    M=len(U[:,0])

    
    # ----------NUMBER OF MODES AND PHOTONS INPUT:----------

    photons=np.zeros(m)

    photons[0]=n


    # ----------UNITARY CHECK FOR MATRIX U:----------


    # Matrix basis generation
    base_u_m, base_u_M, base_u_m_e, base_u_m_f, separator_e_f, base_U_m, base_U_M = matrix_u_basis_generator(m,M,photons,base_input)

    # We obtain the equation system
    eq_sys,eq_sys_choice,index_choice=eq_sys_finder(base_u_m,base_u_M)

    # Verification of the system's validity: in case it is computable, the solution is obtained
    # In case it is not, "None" is given instead
    sol,sol_e,sol_f,check_sol=verification(U,base_u_m,base_u_m_e,base_u_m_f,separator_e_f,base_u_M,eq_sys,eq_sys_choice,index_choice)



    # ----------PART 2: S MATRIX OBTENTION:----------

    # In case a solution exists, S is rebuilt with the given results
    if check_sol==True:
        S=S_output(base_u_m,base_U_m,sol_e,sol_f)


    return S



if __name__ == '__main__':
    modes = 4
    photons = 2
    S = qop.symmetric_map(modes,photons)
    Unl = S @ qut.mkron( qog.random_unitary(modes), photons) @ np.transpose(S)


    U = SfromU(U=Unl, m=modes, n=photons)

    plt.imshow(np.real(U @ qut.dagger(U)))
    plt.show()