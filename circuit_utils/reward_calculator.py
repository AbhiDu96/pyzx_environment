import fractions
import copy
import pyzx as zx

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
    for k in phases.keys():
        phase=phases[k]
        if isinstance(phase, fractions.Fraction):
            if not phase.numerator==0:
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
            print(type(phase))
            assert 0, 'phases must be fractions or integers'
    return tcount


def check_equality(g1,g2):
    g_combined=g1+g2.adjoint()
    zx.full_reduce(g_combined)
    # next line: if a graph corresponds to the identity it has the same number of edges as qubits
    # and correspondingly as inputs
    if (len(g_combined.inputs())==len(list(g_combined.edges()))):
        return True
    else:
        return False
