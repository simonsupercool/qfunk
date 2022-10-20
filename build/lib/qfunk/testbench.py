import qfunk.utility as qut
import qfunk.qoptic as qop
from thewalrus import perm

import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import block_diag

# import Strawberry Fields
import strawberryfields as sf
from strawberryfields.ops import *

np.random.seed(42)

# initialize a 4 mode program
boson_sampling = sf.Program(4)

with boson_sampling.context as q:
    # prepare the input fock states
    Fock(1) | q[0]
    Fock(1) | q[1]
    Vac     | q[2]
    Fock(1) | q[3]

    # rotation gates
    Rgate(0.5719)  | q[0]
    Rgate(-1.9782) | q[1]
    Rgate(2.0603)  | q[2]
    Rgate(0.0644)  | q[3]

    # beamsplitter array
    BSgate(0.7804, 0.8578)  | (q[0], q[1])
    BSgate(0.06406, 0.5165) | (q[2], q[3])
    BSgate(0.473, 0.1176)   | (q[1], q[2])
    BSgate(0.563, 0.1517)   | (q[0], q[1])
    BSgate(0.1323, 0.9946)  | (q[2], q[3])
    BSgate(0.311, 0.3231)   | (q[1], q[2])
    BSgate(0.4348, 0.0798)  | (q[0], q[1])
    BSgate(0.4368, 0.6157)  | (q[2], q[3])


eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 7})


results = eng.run(boson_sampling)


# extract the joint Fock probabilities
probs = results.state.all_fock_probs()

# print the joint Fock state probabilities
print(probs[1, 1, 0, 1])
print(probs[2, 0, 0, 1])


Uphase = np.diag([np.exp(0.5719*1j),np.exp(-1.9782*1j),np.exp(2.0603*1j),np.exp(0.0644*1j)])

BSargs = [
    (0.7804, 0.8578),
    (0.06406, 0.5165),
    (0.473, 0.1176),
    (0.563, 0.1517),
    (0.1323, 0.9946),
    (0.311, 0.3231),
    (0.4348, 0.0798),
    (0.4368, 0.6157)
]

t_r_amplitudes = [(np.cos(q), np.exp(p*1j)*np.sin(q)) for q,p in BSargs]


BSunitaries = [np.array([[t, -np.conj(r)], [r, t]]) for t,r in t_r_amplitudes]

UBS1 = block_diag(*BSunitaries[0:2])
UBS2 = block_diag([[1]], BSunitaries[2], [[1]])
UBS3 = block_diag(*BSunitaries[3:5])
UBS4 = block_diag([[1]], BSunitaries[5], [[1]])
UBS5 = block_diag(*BSunitaries[6:8])

U = multi_dot([UBS5, UBS4, UBS3, UBS2, UBS1, Uphase])
print(np.round(U, 4))

prog_unitary = sf.Program(4)
prog_unitary.circuit = boson_sampling.circuit[4:]
prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")

if __name__ == '__main__':
    qop.symmetric_map(3,3)
    exit()