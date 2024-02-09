import pyzx as zx
from pyzx.utils import EdgeType, VertexType, FractionLike
from pyzx.utils import toggle_edge, vertex_is_zx, toggle_vertex
from pyzx.graph.base import BaseGraph, VT, ET
from typing import Callable, Optional, List, Tuple,Dict,TypeVar
from pyzx import rules
from fractions import Fraction
import random
RewriteOutputType = Tuple[Dict[ET,List[int]], List[VT], List[ET], bool]
MatchObject = TypeVar('MatchObject')



def apply_rule(
        g: BaseGraph[VT,ET], 
        rewrite: Callable[[BaseGraph[VT,ET], List[MatchObject]],RewriteOutputType[ET,VT]],
        m: List[MatchObject], 
        check_isolated_vertices:bool=True
        ) -> None:
    etab, rem_verts, rem_edges, check_isolated_vertices = rewrite(g, m)
    g.add_edge_table(etab)
    g.remove_edges(rem_edges)
    g.remove_vertices(rem_verts)
    if check_isolated_vertices: g.remove_isolated_vertices()




### Soft rules ########################

def matchU_X_spiders(
        g: BaseGraph[VT, ET],
        vertexf: Optional[Callable[[VT], bool]] = None
        ) -> List[VT]:
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    types = g.types()

    return [v for v in candidates if types[v] == VertexType.X]


def matchU_Z_spiders(
        g: BaseGraph[VT, ET],
        vertexf: Optional[Callable[[VT], bool]] = None
        ) -> List[VT]:
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    types = g.types()

    return [v for v in candidates if types[v] == VertexType.Z]


def match_bialgebra(g: BaseGraph[VT,ET], 
        edgef: Optional[Callable[[ET],bool]] = None
        ) -> List[Tuple[VT,VT]]:
    if edgef is not None: candidates = set([e for e in g.edges() if edgef(e)])
    else: candidates = g.edge_set()
    m = []
    types = g.types()
    phases = g.phases()
    while len(candidates) > 0:
        e = candidates.pop()
        if g.edge_type(e) != EdgeType.SIMPLE: continue
        v,w = g.edge_st(e)
        if types[v] != VertexType.X:
            v,w = w,v
        if types[v] != VertexType.X: continue
        if types[w] == VertexType.Z:
            
            # added the next line to remove trivial applications
            if len(g.neighbors(v))<=2 or len(g.neighbors(w))<=2: continue
            
            #I removed next line because I introuced an additional unfusion step in the bialgebra method
            #if phases[v] != 0 or phases[w] != 0: continue
            m.append((v,w))
            # I removed the next lines. We want all matches not only noninteracting ones. This
            # is not a problem because we always only apply one rule at a time.
            #for n in g.neighbors(v):
            #    candidates.difference_update(g.incident_edges(n))
            #for n in g.neighbors(w):
            #    candidates.difference_update(g.incident_edges(n))
            
        # can be removed. We are not considering Hboxes  
        #if types[w] == VertexType.H_BOX:
        #    if phases[v] != 0 or phases[w] != 1: continue
        #    m.append((v,w))
        #    for n in g.neighbors(v):
         #       candidates.difference_update(g.incident_edges(n))
         #   for n in g.neighbors(w):
         #       candidates.difference_update(g.incident_edges(n))
    return m


def bialgebra(g, match):
    assert isinstance(match, tuple) and len(match) == 2, 'wrong input format bialgebra'
    matches=[match]
    rem_verts = []
    etab = {}

    for v,w in matches:
        #############################
        # I added the following lines. In the original implementation of the matcher function,
        # spiders with phases are excluded. I included them in the matcher function. As a result we first 
        # have to unfuse them.
        
        neighbors=list(g.neighbors(v))
        # remove the neighbor w s.t. unfusion is not done in the direction of the bialgebra rule.
        # Take any other neighbor, here the first element of the neighbors
        neighbors.remove(w)
        neighbor=neighbors[0]
        to_unpider=[v, [neighbor]]
        unspider(g,to_unpider)
        
        neighbors=list(g.neighbors(w))
        # remove the neighbor w s.t. unfusion is not done in the direction of the bialgebra rule.
        # Take any other neighbor, here the first element of the neighbors
        neighbors.remove(v)
        neighbor=neighbors[0]
        to_unpider=[w, [neighbor]]
        unspider(g,to_unpider)

        ###################################
        
        rem_verts.append(v)
        rem_verts.append(w)
        new_verts = []
        # v is an X-spider, but w is either a Z-spider or an H-box
        t = g.type(w)
        for n in g.neighbors(v):
            if n == w: continue
            r = 0.6*g.row(v) + 0.4*g.row(n)
            q = 0.6*g.qubit(v) + 0.4*g.qubit(n)
            v2 = g.add_vertex(t,q,r)
            etab[g.edge(n,v2)] = [1,0] if g.edge_type(g.edge(n,v)) == EdgeType.SIMPLE else [0,1]
            new_verts.append(v2)
        if g.type(w) == VertexType.Z:
            t = VertexType.X
            g.scalar.add_power((g.vertex_degree(v)-2)*(g.vertex_degree(w)-2))
        else: #g.type(w) == VertexType.H_BOX
            t = VertexType.Z
            g.scalar.add_power(g.vertex_degree(v)-2)
        for n in g.neighbors(w):
            if n == v: continue
            r = 0.6*g.row(w) + 0.4*g.row(n)
            q = 0.6*g.qubit(w) + 0.4*g.qubit(n)
            w2 = g.add_vertex(t,q,r)
            etab[g.edge(n,w2)] = [1,0] if g.edge_type(g.edge(n,w)) == EdgeType.SIMPLE else [0,1]
            for v2 in new_verts:
                etab[g.edge(w2,v2)] = [1,0]
    return (etab, rem_verts, [], False)


def match_euler(
        g: BaseGraph[VT,ET], 
        edgef: Optional[Callable[[ET],bool]] = None
        ) -> List[ET]:
    if edgef is not None: candidates = set([e for e in g.edges() if edgef(e)])
    else: candidates = g.edge_set()
    return [e for e in candidates if g.edge_type(e)==EdgeType.HADAMARD]


def euler(g, match):
    assert isinstance(match, tuple) and len(match) == 2, 'wrong input format euler'
    matches=[match]
    types = g.types()
    phases = g.phases()
    rem_edges = []
    etab = {}
    for e in matches:
        rem_edges.append(e)
        v1,v2 = g.edge_st(e)
        if vertex_is_zx(types[v1]) and types[v1] == types[v2]:
            r = 0.5*(g.row(v1) + g.row(v2))
            q = 0.5*(g.qubit(v1) + g.qubit(v2))
            t = toggle_vertex(types[v1])
            v = g.add_vertex(t,q,r)
            etab[g.edge(v,v1)] = [1,0]
            etab[g.edge(v,v2)] = [1,0]
            if phases[v1] == Fraction(1,2) or phases[v2] == Fraction(1,2):
                g.add_to_phase(v1,Fraction(3,2))
                g.add_to_phase(v2,Fraction(3,2))
                g.set_phase(v, Fraction(3,2))
                g.scalar.add_phase(Fraction(1,4))
            else:
                g.add_to_phase(v1,Fraction(1,2))
                g.add_to_phase(v2,Fraction(1,2))
                g.set_phase(v, Fraction(1,2))
                g.scalar.add_phase(Fraction(7,4))
        else:
            r = 0.25*g.row(v1) + 0.75*g.row(v2)
            q = 0.25*g.qubit(v1) + 0.75*g.qubit(v2)
            w1 = g.add_vertex(VertexType.Z,q,r,Fraction(1,2))
            etab[g.edge(v2,w1)] = [1,0]
            r = 0.5*g.row(v1) + 0.5*g.row(v2)
            q = 0.5*g.qubit(v1) + 0.5*g.qubit(v2)
            w2 = g.add_vertex(VertexType.X,q,r,Fraction(1,2))
            etab[g.edge(w1,w2)] = [1,0]
            r = 0.75*g.row(v1) + 0.25*g.row(v2)
            q = 0.75*g.qubit(v1) + 0.25*g.qubit(v2)
            w3 = g.add_vertex(VertexType.Z,q,r,Fraction(1,2))
            etab[g.edge(w2,w3)] = [1,0]
            etab[g.edge(w3,v1)] = [1,0]
            g.scalar.add_phase(Fraction(7,4))
            
    return (etab, [], rem_edges, False)


def match_color_change(
        g: BaseGraph[VT, ET],
        vertexf: Optional[Callable[[VT], bool]] = None
        ) -> List[VT]:
    '''Matches x and z spiders '''
    mx=matchU_X_spiders(g,vertexf)
    mz=matchU_Z_spiders(g,vertexf)
    return mx+mz


def color_change(g: BaseGraph[VT,ET], match) -> rules.RewriteOutputType[ET,VT]:
    assert isinstance(match, int), 'wrong input format color change'
    matches=[match]
    for v in matches:
        g.set_type(v, toggle_vertex(g.type(v)))
        for e in g.incident_edges(v):
            et = g.edge_type(e)
            g.set_edge_type(e, toggle_edge(et))
    return ({}, [],[],False)


def match_pi_copy(
        g: BaseGraph[VT,ET], 
        vertexf: Optional[Callable[[VT],bool]] = None
        ) -> List[Tuple[VT,VT]]:
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    phases = g.phases()
    types = g.types()
    m: List[Tuple[VT,VT]] = []
    paulis = {v for v in candidates
                if phases[v] == 1 and vertex_is_zx(types[v])}
    if not paulis: return m
    
    # attention! The elemnts in m (w,v) are not general edges but ordered, that is,
    # w is the Pauli and v the vertex we push it through
    
    while len(candidates) > 0:    
        v = candidates.pop()
        # The next line prohibits pushing a Pauli gate through a Pauli gate. But why would
        # we want to prohibit this?
        # if v in paulis and g.vertex_degree(v) == 2: continue
        for w in g.neighbors(v):
            if w in paulis: break
        else:
            continue
        et = g.edge_type(g.edge(v,w))
        if ((types[v] == types[w] and et == EdgeType.HADAMARD) or
            (vertex_is_zx(types[v]) and types[v] != types[w] and et == EdgeType.SIMPLE) or
            (types[v] == VertexType.H_BOX and phases[v] == 1 and (
                (et == EdgeType.SIMPLE and types[w] == VertexType.X) or
                (et == EdgeType.HADAMARD and types[w] == VertexType.Z)))
            ):
            m.append((w,v))
            
            #candidates.difference_update(g.neighbors(v))
            #candidates.difference_update(g.neighbors(w))
            
 
            
    return m


def pi_copy(g, match):
    assert isinstance(match, tuple) and len(match) == 2, 'wrong input format pi_copy'
    rem_verts = []
    rem_edges = []
    etab = {}
    matches=[match]
    for w,v in matches:  # w is a Pauli and v is the spider we are gonna push it through
        if g.vertex_degree(w) == 2:
            rem_verts.append(w)
            l = list(g.neighbors(w))
            l.remove(v)
            v2 = l[0]
            et1 = g.edge_type(g.edge(v,w))
            et2 = g.edge_type(g.edge(v2,w))
            etab[g.edge(v,v2)] = [1,0] if et1 == et2 else [0,1]
        else:
            g.set_phase(w,0)
        new_verts = []
        if vertex_is_zx(g.type(v)): 
            g.scalar.add_phase(g.phase(v))
            g.set_phase(v,(-g.phase(v)) % 2)
            t = toggle_vertex(g.type(v))
            p: FractionLike = Fraction(1)
        else: 
            t = VertexType.Z
            p = 0
        for n in g.neighbors(v):
            if n == w: continue
            r = 0.5*(g.row(n) + g.row(v))
            q = 0.5*(g.qubit(n) + g.qubit(v))
            e = g.edge(n,v)
            et = g.edge_type(e)
            rem_edges.append(e)
            w2 = g.add_vertex(t,q,r,p)
            etab[g.edge(v,w2)] = [1,0]
            etab[g.edge(n,w2)] = [1,0] if et == EdgeType.SIMPLE else [0,1]
            new_verts.append(w2)
        if not vertex_is_zx(g.type(v)): # v is H_BOX
            if len(new_verts) == 2:
                etab[g.edge(new_verts[0],new_verts[1])] = [0,1]
            else:
                r = (g.row(v) + sum(g.row(n) for n in new_verts)) / (len(new_verts) + 1)  # type: ignore # I don't understand this error
                q = (g.qubit(v) + sum(g.qubit(n) for n in new_verts))/(len(new_verts)+1)  # type: ignore
                h = g.add_vertex(VertexType.H_BOX,q,r,Fraction(1))
                for n in new_verts: etab[g.edge(h,n)] = [1,0]
    return (etab, rem_verts, rem_edges, False)


def match_add_identity(
        g: BaseGraph[VT,ET], 
        edgef: Optional[Callable[[ET],bool]] = None
        ) -> List[ET]:
    if edgef is not None: candidates = set([e for e in g.edges() if edgef(e)])
    else: candidates = g.edge_set()
    return list(candidates)


def add_identity(g: BaseGraph[VT,ET], 
        match
        ) -> rules.RewriteOutputType[ET,VT]:
    assert isinstance(match, tuple) and len(match) == 2, 'wrong input format add_identity'
    rem_edges = []
    etab = {}
    matches=[match]
    for e in matches:
        rem_edges.append(e)
        et = g.edge_type(e)
        v1,v2 = g.edge_st(e)
        r = 0.5*(g.row(v1) + g.row(v2))
        q = 0.5*(g.qubit(v1) + g.qubit(v2))
        w = g.add_vertex(VertexType.Z, q,r, 0)
        etab[g.edge(v1,w)] = [1,0] if et == EdgeType.SIMPLE else [0,1]
        etab[g.edge(v2,w)] = [1,0]
    return (etab, [], rem_edges, False)


def match_add_hadamard_identity(g, edgef=None):
    # only simple edges (no Hadamard edges are allowed)
    allowed_edges=[e for e in g.edges() if g.edge_type(e)==EdgeType.SIMPLE ]
    if edgef is not None: candidates = set([e for e in allowed_edges if edgef(e)])
    else: candidates = allowed_edges
    return list(candidates)


def add_hadamard_identity(g,match):
    assert isinstance(match, tuple) and len(match) == 2, 'wrong input format add_hadamard_identity'
    rem_edges = []
    etab = {}
    vertices_added=[]
    matches=[match]
    for e in matches:
        rem_edges.append(e)
        v1,v2 = g.edge_st(e) # v1,v2 Ränder der edges e
        r = 0.5*(g.row(v1) + g.row(v2))
        q = 0.5*(g.qubit(v1) + g.qubit(v2))
        w = g.add_vertex(VertexType.Z, q,r, 0)
        etab[g.edge(v1,w)] = [0,1] 
        etab[g.edge(w,v2)] = [0,1]
    return (etab, [], rem_edges, False)


def match_spider_fusion(
        g: BaseGraph[VT,ET], 
        matchf:Optional[Callable[[ET],bool]]=None
        ): 
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(e)])
    else: candidates = g.edge_set()
    types = g.types()

    m = []
    for e in candidates:
        if g.edge_type(e) != EdgeType.SIMPLE: continue
        v0, v1 = g.edge_st(e)
        v0t = types[v0]
        v1t = types[v1]
        if (v0t == v1t and vertex_is_zx(v0t)):
                m.append((v0,v1))
    return m


def spider_fusion(g: BaseGraph[VT,ET], match) -> RewriteOutputType[ET,VT]:
    assert isinstance(match, tuple) and len(match) == 2, 'wrong input format spider fusion'
    matches=[match]
    rem_verts = []
    etab: Dict[ET,List[int]] = dict()
    types = g.types()

    for m in matches:
        if g.row(m[0]) == 0:
            v0, v1 = m[1], m[0]
        else:
            v0, v1 = m[0], m[1]

        ground = g.is_ground(v0) or g.is_ground(v1)
        if ground:
            g.set_phase(v0, 0)
            g.set_ground(v0)
        else:
            g.add_to_phase(v0, g.phase(v1))

        if g.track_phases:
            g.fuse_phases(v0,v1)

        # always delete the second vertex in the match
        rem_verts.append(v1)

        # edges from the second vertex are transferred to the first
        for w in g.neighbors(v1):
            if v0 == w: continue
            e = g.edge(v0,w)
            if e not in etab: etab[e] = [0,0]
            etab[e][g.edge_type(g.edge(v1,w))-1] += 1
    return (etab, rem_verts, [], True)


def match_unspider(
        g: BaseGraph[VT, ET],
        vertexf: Optional[Callable[[VT], bool]] = None
        ) -> List[VT]:
    
    mZ=matchU_Z_spiders(g, vertexf)
    mX=matchU_X_spiders(g, vertexf)
    return mZ+mX


def unspider(g, m, qubit=-1,  row=-1):
    """Undoes a single spider fusion, given a match ``m``. A match is a list with 3
    elements given by::

      m[0] : a vertex to unspider
      m[1] : a list. m[1] is either [], then the phase is put into a new node without connections to others
             or a list with one element corresponding to the vertex, the phase is pushed towards
      m[2] : the phase of the new node. If omitted, the new node gets all of the phase of m[0]

    Returns the index of the new node. Optional parameters ``qubit`` and ``row`` can be used
    to position the new node. If they are omitted, they are set as the same as the old node.
    """
    assert len(m[1])==0 or len(m[1])==1, 'wrong input format unspider'
    u = m[0] 
    
    # only plotting
    if len(m[1])==0:
        r = g.row(u)
        if len(g.inputs())>=1:
            delta=abs(g.qubit(g.inputs()[0])-g.qubit(g.inputs()[1]))
            q = g.qubit(u)+delta/2
        else:
            q = g.qubit(u)
    elif len(m[1])==1:
        r = 0.5*(g.row(u) + g.row(m[1][0]))
        q = 0.5*(g.qubit(u) + g.qubit(m[1][0]))
    else:
        assert 0, 'number of vertices neighbors must either be zero or one' 
      
    v = g.add_vertex(g.type(u),q,r)
    u_is_ground = g.is_ground(u)

    g.add_edge(g.edge(u, v))
    
    for n in m[1]:
        e = g.edge(u,n)
        g.add_edge(g.edge(v,n), edgetype=g.edge_type(e))
        g.remove_edge(e)
    if len(m) >= 3:
        g.add_to_phase(v, m[2])
        if not u_is_ground:
            g.add_to_phase(u, Fraction(0) - m[2])
    else:
        g.set_phase(v, g.phase(u))
        g.set_phase(u, 0)
    return v


### Hard rules ########################

def full_fuse(g):
    zx.spider_simp(g, quiet=True)
    
def full_id_remove(g):
    zx.id_simp(g, quiet=True) 
