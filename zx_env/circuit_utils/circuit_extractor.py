import pyzx as zx
from pyzx import rules
from pyzx.simplify import id_simp, spider_simp
from pyzx.simplify import to_graph_like, spider_simp, to_gh,lcomp_simp,simp
import copy
from pyzx.utils import VertexType, EdgeType, toggle_edge, vertex_is_zx, FloatInt, FractionLike
from fractions import Fraction



def extract_circuit(graph, quiet=True):
    '''The new circuit extraction method makes use of graph-like states, the local complementation
       rule as well as the pivoting rule. It tries to use these rules as carefully as possibly to avoid
       introducing too many CX gates.
       To guarantee the extraction of a circuit, it proceeds the following way:
       - level 1: It removes all Clifford gates (in particular pi/2 and pi) 
       that have at most 2 neighbours.
       By that procedure, the pi-copy rule and Hadamard decomposition rule is undone.
       - Tries to extract the circuit. If unsuccessful:
       - level 2: Pivotes along every edge if at most one vertex has more neighbors.
       - Tries to extract the circuit. If unsuccessful:
       - level 3: Applies every possible local complementation rule
       - Tries to extract the circuit. If unsuccessful:
       - level 4: Applies every pivoting and every possible local complementation rule.
       - Tries to extract the circuit. If unsuccessful:
       - level 5: Gives up and uses full_reduce '''
    g=copy.deepcopy(graph)
    g.normalize()
    to_graph_like(g)
    g.normalize()
    
    for relax_level in [1,2,3,4]:
        interior_clifford_simp_on_wire(g, relax_level, quiet=quiet, stats=None)
        try:
            return (zx.extract_circuit(copy.deepcopy(g), up_to_perm=False), relax_level)
        except:
            if relax_level==4:
                zx.full_reduce(g)
                return (zx.extract_circuit(g, up_to_perm=False), 5)
            





def match_pivot_on_wire1(g, matchf=None, num=-1, check_edge_types=True):
    return match_pivot_on_wire(g, level=1, matchf=matchf, num=num, check_edge_types=check_edge_types) 

def match_pivot_on_wire2(g, matchf=None, num=-1, check_edge_types=True):
    return match_pivot_on_wire(g, level=2, matchf=matchf, num=num, check_edge_types=check_edge_types)

def match_pivot_on_wire3(g, matchf=None, num=-1, check_edge_types=True):
    return match_pivot_on_wire(g, level=3, matchf=matchf, num=num, check_edge_types=check_edge_types)
    


def pivot_simp_on_wire(g, level=1, matchf=None, quiet=False, stats=None):
    if level==1:
        return simp(g, 'pivot_simp', match_pivot_on_wire1, rules.pivot, matchf=matchf, quiet=quiet, stats=stats)
    if level==2:
        return simp(g, 'pivot_simp', match_pivot_on_wire2, rules.pivot, matchf=matchf, quiet=quiet, stats=stats)
    if level==3:
        return simp(g, 'pivot_simp', match_pivot_on_wire3, rules.pivot, matchf=matchf, quiet=quiet, stats=stats)
        
    

def match_lcomp_on_wire1(g, level=1,vertexf=None, num=-1, check_edge_types=True):
    return  match_lcomp_on_wire(g, level=1,vertexf=vertexf, num=num, check_edge_types=check_edge_types)

def match_lcomp_on_wire2(g, level=1,vertexf=None, num=-1, check_edge_types=True):
    return  match_lcomp_on_wire(g, level=2,vertexf=vertexf, num=num, check_edge_types=check_edge_types)


def lcomp_simp_on_wire(g,level=1, matchf=None, quiet=False, stats=None):
    if level==1:
        return simp(g, 'lcomp_simp', match_lcomp_on_wire1, rules.lcomp, matchf=matchf, quiet=quiet, stats=stats)
    if level==2:
        return simp(g, 'lcomp_simp', match_lcomp_on_wire2, rules.lcomp, matchf=matchf, quiet=quiet, stats=stats)




def interior_clifford_simp_on_wire(g, relax_level=1, quiet=False, stats=None):
    """Keeps doing the simplifications ``id_simp``, ``spider_simp``,
    ``pivot_simp`` and ``lcomp_simp`` until none of them can be applied anymore."""
    spider_simp(g, quiet=quiet, stats=stats)
    to_gh(g)
    i = 0
    if relax_level==1:
        level_pivot=1
        level_lcomp=1
        
    if relax_level==2:
        level_pivot=2
        level_lcomp=1
        
    if relax_level==3:
        level_pivot=2
        level_lcomp=2
        
    if relax_level==4:
        level_pivot=3
        level_lcomp=2
        
    while True:
        i1 = id_simp(g, quiet=quiet, stats=stats)
        i2 = spider_simp(g, quiet=quiet, stats=stats)
        i3 = pivot_simp_on_wire(g, level_pivot, quiet=quiet, stats=stats)
        i4 = lcomp_simp_on_wire(g, level_lcomp, quiet=quiet, stats=stats)
        if i1+i2+i3+i4==0: break
        i += 1
    return i


def match_lcomp_on_wire(g, level=1,vertexf=None, num=-1, check_edge_types=True):
    """Finds noninteracting matchings of the local complementation rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
       are of the correct type (Hadamard edges).
    :param vertexf: An optional filtering function for candidate vertices, should
       return True if a vertex should be considered as a match. Passing None will
       consider all vertices.
    :rtype: List of 2-tuples ``(vertex, neighbors)``.
    """
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    types = g.types()
    phases = g.phases()
    i = 0
    m = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        
        #### I added this line to only allow spiders with at most two neighbors
        if level==1:
            if len(g.neighbors(v))>2: continue
        elif level==2:
            pass
        else:
            raise ValueError('level must be either 1 or 2')
        #####
        
        vt = types[v]
        va = g.phase(v)
        if vt != VertexType.Z: continue
        if not (va == Fraction(1,2) or va == Fraction(3,2)): continue
        if g.is_ground(v):
            continue
        if check_edge_types and not (
            all(g.edge_type(e) == EdgeType.HADAMARD for e in g.incident_edges(v))
            ): continue          
        vn = list(g.neighbors(v))
        if not all(types[n] == VertexType.Z for n in vn): continue
        for n in vn: candidates.discard(n)
        m.append((v,vn))
    return m



def match_pivot_on_wire(g, level=1, matchf=None, num=-1, check_edge_types=True): 
    """Finds non-interacting matchings of the pivot rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
       are of the correct type (Hadamard edges).
    :param matchf: An optional filtering function for candidate edge, should
       return True if a edge should considered as a match. Passing None will
       consider all edges.
    :rtype: List of 4-tuples. See :func:`pivot` for the details.
    """
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    types = g.types()
    phases = g.phases()
    
    i = 0
    m = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        if check_edge_types and g.edge_type(e) != EdgeType.HADAMARD: continue
        v0, v1 = g.edge_st(e)
        
        #### I added these lines to only allow spiders with at most two neighbors
        if level==1:
            if len(g.neighbors(v0))>2 or len(g.neighbors(v1))>2 : continue
        elif level==2:
            if len(g.neighbors(v0))>2 and len(g.neighbors(v1))>2 : continue
        elif level==3:
            pass
        else:
            raise ValueError('level must be between 1 and 3')
        #####

        if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z): continue

        v0a = phases[v0]
        v1a = phases[v1]
        if not ((v0a in (0,1)) and (v1a in (0,1))): continue
        if g.is_ground(v0) or g.is_ground(v1):
            continue

        invalid_edge = False

        v0n = list(g.neighbors(v0))
        v0b = []
        for n in v0n:
            et = g.edge_type(g.edge(v0,n))
            if types[n] == VertexType.Z and et == EdgeType.HADAMARD: pass
            elif types[n] == VertexType.BOUNDARY: v0b.append(n)
            else:
                invalid_edge = True
                break

        if invalid_edge: continue

        v1n = list(g.neighbors(v1))
        v1b = []
        for n in v1n:
            et = g.edge_type(g.edge(v1,n))
            if types[n] == VertexType.Z and et == EdgeType.HADAMARD: pass
            elif types[n] == VertexType.BOUNDARY: v1b.append(n)
            else:
                invalid_edge = True
                break

        if invalid_edge: continue
        if len(v0b) + len(v1b) > 1: continue

        i += 1
        for v in v0n:
            for c in g.incident_edges(v): candidates.discard(c)
        for v in v1n:
            for c in g.incident_edges(v): candidates.discard(c)
        b0 = list(v0b)
        b1 = list(v1b)
        m.append((v0,v1,b0,b1))
    return m
