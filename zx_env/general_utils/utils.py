import pyzx as zx
import fractions
import copy
from pyzx import EdgeType

"""def check_equality(g1,g2):
    g_combined=g1+g2.adjoint()
    zx.full_reduce(g_combined)
    # next line: if a graph corresponds to the identity it has the same number of edges as qubits
    # and correspondingly as inputs
    if (len(g_combined.inputs())==len(list(g_combined.edges()))):
        return True
    else:
        return False
"""
def check_equality(g1,g2):
    '''Function checks if two graphs g1 and g2 are the same. It combines g1 with the adjoint of g2 and then 
    applies full_reduce. If the resulting graph only contains input and output nodes and the conneting lines are not 
    Hadamard edges, it returnes True, otherwise False. Note, however, that the result False does not mean that the graphs 
    are not the same but only that the reduction did not succeed. Further note, that if g1=g2 up to permutations, the function
    still returns True.'''
    g_combined=copy.deepcopy(g1)+copy.deepcopy(g2.adjoint())
    zx.full_reduce(g_combined)
    # next line: if a graph corresponds to the identity it has the same number of edges as qubits
    # and correspondingly as inputs
    if (len(g_combined.inputs())==len(list(g_combined.edges()))):
        return EdgeType.HADAMARD not in [g_combined.edge_type(e) for e in g_combined.edges()]
    else:
        return False


def tcount_from_graph(g):
    '''Function calculates an approximation to the expected number of T gates for the extracted circuit
       given a graph. If the circuit can be extracted, the approximation is exact. The functions throws an
       error if the circuit does not consist of Clifford+T gates only.'''
    tcount=0
    g_copy=copy.deepcopy(g)
    # first fuse all possible spiders and remove all identities
    while True:
        r1=zx.id_simp(g_copy, quiet=True) 
        r2=zx.spider_simp(g_copy, quiet=True)
        if r1==0 and r2==0:
            break
    # go through all phases. They must be integer multiples of pi/4
    phases=g_copy.phases()
    for k in phases:
        phase=phases[k]
        if isinstance(phase, fractions.Fraction):
            if phase.numerator != 0:
                short_denom=(phase.limit_denominator()).denominator
                if short_denom==4:
                    tcount+=1
                elif short_denom in [1,2]:
                        pass
                else:
                    assert 0, 'phases must be integer multiples of pi/4'
        elif isinstance(phase, int):
            pass
        else:
            import logging; logging.warning("unexpected phase type: %s", type(phase))  # noqa: I001, LOG015
            assert 0, 'phases must be fractions or integers'
    return tcount

