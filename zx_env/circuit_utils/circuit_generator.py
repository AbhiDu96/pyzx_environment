import numpy as np
import pyzx as zx
from fractions import Fraction

def random_circuit(n_qubit=5, num_gates=40, p_two_qubit=0.1, p_H=0.7, p_z=0.1, p_x=0.1, many_pi_gates=False,clifford_plus_T=False, 
    add_cz=False):
    '''Sample a random circuit
       n_qubit: number of qubits
       num_gates: total number of gates in the circuit
       p_two_qubit: probability for a CNOT
       p_H: probabbility for a Hadamard gate
       p_z: probability for a z rotation
       p_x: probability for an x rotation
       if many_pi_gates==True, we replace the phases of 50% of the nodes by pi to. This is required 
       to check the correctnes of the pi_copy rule
       clifford_plus_T: If true only Clifford+T gates are applied'''
    c = zx.Circuit(qubit_amount=n_qubit)
    for i in range(num_gates):
        gate=np.random.choice(range(4),p=[p_two_qubit,p_z,p_x,p_H])
        if gate==0: # add CNOT gate
            target=np.random.choice(range(n_qubit))
            while True:
                control=np.random.choice(range(n_qubit))
                if not control==target:
                    break
            two_qubit_gate=np.random.choice(range(2))
            if two_qubit_gate==0 and add_cz:
                c.add_gate("CZ",control,target)
            else:
                c.add_gate("CNOT",control,target)
                
        elif gate==1: # add Hadamard gate
            qubit=np.random.choice(range(n_qubit))
            c.add_gate("H", qubit) 
        elif gate==2: # add z-spider
            qubit=np.random.choice(range(n_qubit))
            a=np.random.choice(range(10))
            if clifford_plus_T:
                b=4
            else:
                b=np.random.choice(range(1,10))
            c.add_gate("ZPhase", qubit, phase=Fraction(a,b))   
        elif gate==3: # add x-spider
            qubit=np.random.choice(range(n_qubit))
            a=np.random.choice(range(10))
            if clifford_plus_T:
                b=4
            else:
                b=np.random.choice(range(1,10))
            c.add_gate("XPhase", qubit, phase=Fraction(a,b))     
    if many_pi_gates:
        return replace_by_pi_gates(c)
    else:
        return c



def replace_by_pi_gates(circuit, p=0.5):
    assert p>=0 and p<=1, 'p must be between 0 and 1'
    for i, gate in enumerate(circuit.gates):
        if gate.name=='XPhase' or gate.name=='ZPhase':
            x=np.random.random()
            if x<=p:
                circuit.gates[i].phase=Fraction(1,1)
    return circuit

