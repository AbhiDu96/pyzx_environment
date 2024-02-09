import pyzx as zx
import qiskit
import time
import copy

class Circuit_extractor():
    '''The Circuit_extractor is a simple implementation of a method that can extract circuits
       from ZX diagrams that are still relatively close to a circuit structure.
       For example the following ZX diagram clearly can be written as a circuit. 
       The output of the circuit extractor would be the circuit shown on the right.
       The following properties characterize the extractor:
       The circuit extractor extracts the graph as a pyzx circuit that can easily be converted
       to a qiskit circuit. The circuit is returned in the gate set: H, CNOT, Rx, Rz.
       It automatically fuses some of the rotation gates, and cancels some of the CNOT gates.
       In addition if several H gates appear after each other, they are canceled. However,
       there might be needed additional sweeps for simplifications. The method first copies
       the attributes of the pyzx GaphS object into a dictionary and works on that.
       I tried to work directly on the graph but it turned out to be much slower. Of course the
       pyzx circuit extraction is much more general but using our own implementation, we know exactly 
       what is going on. Furthermore, I suppose our function is faster because it does not use the
       Gauss-Jordan elimination step. '''
    
    def __init__(self,g):  
        '''Initialize the Circuit_extractor object.
           INPUTS:
               GraphS: g: pyzx graph to be extracted
           OUTPUTS: -- '''
        g_copy=g.copy()
        
        for i in g.inputs():
            assert len(g.neighbors(i))==1, 'inputs must not be connected to more than one vertex'
        for o in g.outputs():
            assert len(g.neighbors(o))==1, 'outputs must not be connected to more than one vertex'
        
        assert len(g.inputs())==len(g.outputs()), 'number of inputs and outputs must be equal'
        
        #Initial simplifications, we remove identities and fuse spiders until
        # no matches are left
        while True:
            r1=zx.id_simp(g_copy, quiet=True) 
            r2=zx.spider_simp(g_copy, quiet=True)
            if r1==0 and r2==0:
                break
    
        edges, properties=get_properties(g_copy)
        self.edges=edges
        self.properties=properties
        self.wait_list=list(g_copy.inputs())
        self.num_qubits=len(g_copy.inputs())
        self.circuit=[]
        
        #needed for ading swaps in the end
        self.output_list=list(g_copy.outputs())
        
        for k in properties.keys():
            if properties[k][1]=='h-box':
                assert 0, 'Circuit must not contain h-boxes but only hadamard edges'
        
    
    def extract_circuit(self):
        '''Main method to extract a circuit. The method works on 
           a representation of the graph in the dictionary 'properties' generated in 
           __init()__. This leads to a faster implementation than directly working on the graphS pyzx 
           object. We start from the vertices given in wait_list. Initially the wait_list is given by
           the input nodes. We now call the function single_qubit_steps(). Starting from the first vertex in
           the wait list it walks along the graph, removing vidited nodes, until it reaches a node which
           is connected to several other nodes (two qubit gates). This process is repeated for all vertices
           in the wait list and the wait_list is updated. In the next step we call the function
           two_qubit_steps(). It extracts (and removes) all connections between elements of the wait_list.
           These two steps are repeated until only the output nodes are left. When gates are extracted,
           they are stored as a string in the list self.circuit. Since every CZ gate is extracted as CX gate, 
           there might appear circuits with many repeated Hadamard gates. We therefor perform a sweep through the
           list to cancel these Hadamard gates by the function sweep_hadamard. Then we pass that list
           to the .to_zx_circuit method. It transforms the circuit list to a pyzx.Circuit object.
           Finally, the function checks if we end up with permuted qubit locations. If so, the function adds
           SWAP gates to redo these permutations. Note that the extraction can therefore add more CNOT gates. 
           This needs carful rethinking if we use the method for routing.'''
        self.success=True
        len_old=len(self.properties)
        while len(self.properties)>len(self.wait_list):
            try:
                self.single_qubit_steps()
                self.two_qubit_steps()
            except:
                self.success=False
                break
            len_new=len(self.properties)
            if len_new==len_old: # -> we are stuck -> graph cannot be written as circuit
                self.success=False
                break
            len_old=len_new
        
        # reduce hadamards which appeared by writing CZ gates as CX gates
        self.circuit=sweep_hadamard(self.circuit, self.num_qubits)
         
        #add swaps in the end if necessary    
        try:
            self.add_swaps()
        except:
            self.success=False    
            
        #walk a few times through the circuit and cancel gates
        #that appear after another
        self.cancel_neigboring_gates(rounds=3) 
            
         
        # convert gate list into pyzx circuit
        self.to_zx_circuit(self.circuit)
        
        if not self.success:
            self.circuit=[]
            self.zx_circuit=None
           
        return self.success
    
    
    def single_qubit_steps(self):
        wait_list_new=[]
        for qubit, node in enumerate(self.wait_list):
            go_on=True
            node_horizontal=node
            while go_on:
                go_on, node_horizontal=self.single_step(node_horizontal, qubit)        
                if not go_on:
                    wait_list_new.append(node_horizontal)
        self.wait_list=wait_list_new
        
    
    def single_step(self,node, qubit):        
        neighbors=self.properties[node][2] 
        num_neighbors=self.properties[node][3]

        if num_neighbors>1: # -> it's a node with connections to other qubits
            return (False, node)
    
        elif num_neighbors==0: # -> it's an output node
            return (False, node)
    
        elif num_neighbors==1: # -> it's a single-qubit gate
            node_new=neighbors[0]
            self.add_single_qubit_gate(node, node_new, qubit)
            
            # remove previous node from property list
            del self.properties[node]
            
            # remove connections to previous node at node_new
            (self.properties[node_new][2]).remove(node)
            self.properties[node_new][3]-=1   
            return (True, node_new)


    def two_qubit_steps(self):
        for qubit1, w in enumerate(self.wait_list):
            connections=self.properties[w][2]
            for  c in connections:
                if c in self.wait_list:
                    qubit2=self.wait_list.index(c)
                    
                    self.add_two_qubit_gate(w, c, qubit1,qubit2)
                    (self.properties[c][2]).remove(w)
                    self.properties[c][3]-=1
                    (self.properties[w][2]).remove(c)
                    self.properties[w][3]-=1
                    
                    

    def add_single_qubit_gate(self, node,node_new, qubit):
        if not self.properties[node][0].numerator==0: # -> only add if phase ~= 0 
            self.circuit.append((self.properties[node][1],self.properties[node][0], qubit))
        if self.edges[(node, node_new)]==zx.utils.EdgeType.HADAMARD:
            self.circuit.append(('H', qubit))

            
    def add_two_qubit_gate(self, node1, node2, qubit1,qubit2):       
        type1=self.properties[node1][1]
        type2=self.properties[node2][1]
        type12=self.edges[(node1,node2)]
        
        if type1=='z-spider' and type2=='x-spider' and type12==zx.utils.EdgeType.SIMPLE:
            self.circuit.append(('CX', (qubit1,qubit2)))
        
        elif type1=='x-spider' and type2=='z-spider' and type12==zx.utils.EdgeType.SIMPLE:
            self.circuit.append(('CX', (qubit2,qubit1)))
        
        elif type1=='x-spider' and type2=='x-spider' and type12==zx.utils.EdgeType.HADAMARD:
            self.circuit.append(('H', qubit2))
            self.circuit.append(('CX', (qubit2,qubit1)))
            self.circuit.append(('H', qubit2))
            
        elif type1=='z-spider' and type2=='z-spider' and type12==zx.utils.EdgeType.HADAMARD:
            self.circuit.append(('H', qubit2))
            self.circuit.append(('CX', (qubit1,qubit2)))
            self.circuit.append(('H', qubit2))
        else:
            self.success=False
            
    def to_qiskit(self):
        c_qasm=self.zx_circuit.to_qasm()
        qc=qiskit.QuantumCircuit.from_qasm_str(c_qasm)
        return qc
                
            
    def to_zx_circuit(self,circuit):
        zx_circuit= zx.Circuit(qubit_amount=self.num_qubits)
        for c in circuit:
            if c[0]=='CX':
                zx_circuit.add_gate("CNOT",c[1][0],c[1][1])
            elif c[0]=='z-spider':
                zx_circuit.add_gate("ZPhase", c[2], phase=c[1])
            elif c[0]=='x-spider':
                zx_circuit.add_gate("XPhase", c[2], phase=c[1])
            elif c[0]=='H':
                zx_circuit.add_gate("H", c[1])
            else:
                assert 0, 'Not implemented error'
        self.zx_circuit=zx_circuit
        
        
    def add_swaps(self):
        if not set(self.output_list)==set(self.wait_list):
            self.success=False
        permutations=get_permutations(copy.deepcopy(self.output_list), copy.deepcopy(self.wait_list))
        self.permutations=permutations
        for swap_index in permutations[::-1]:
            # append the final SWAP gate such that it is more likely some gates can be canceled
            final_gate=self.circuit[-1]
            if final_gate[0]=="CX" and final_gate[1]==swap_index:
                self.circuit.append(("CX", swap_index))
                self.circuit.append(("CX", (swap_index[1],swap_index[0])))
                self.circuit.append(("CX", swap_index))
            else:
                self.circuit.append(("CX", (swap_index[1],swap_index[0])))
                self.circuit.append(("CX", swap_index))
                self.circuit.append(("CX", (swap_index[1],swap_index[0])))
                    
  
    def cancel_neigboring_gates(self,rounds):
        for _ in range(rounds):
            i=0
            while i < len(self.circuit) - 1:
                if self.circuit[i] == self.circuit[i+1]:
                    # later fixed this bug. We can only cancel neighboring gates if they are not
                    # z-spiders or x-spiders (that is H gates or CX gates)
                    if not self.circuit[i][0] in ['z-spider', 'x-spider']:
                        del self.circuit[i]
                        del self.circuit[i]
                    else:
                        i+=1
                else:
                    i += 1

                    
                    
def get_properties(g):
    # create a dictionarycontaining the vertices, its type and phases
    properties={}
    inputs=g.inputs()
    for node in list(g.phases().keys()):
        if g.type(node)==zx.utils.VertexType.BOUNDARY:
            if node in inputs:
                edge_type='input'
            else:
                edge_type='output'
        elif g.type(node)==zx.utils.VertexType.Z:
            edge_type='z-spider'
        elif g.type(node)==zx.utils.VertexType.X:
            edge_type='x-spider'
        elif g.type(node)==zx.utils.VertexType.H_BOX:
            edge_type='h-box'
        neighbors=list(g.neighbors(node))
        properties[node]=[g.phases()[node],edge_type,neighbors, len(neighbors)]
        
    # edges are included as tuples
    edges={}
    for e in g.edge_set():
        edges[e]=g.edge_type(e) 
    # add edges both directions, i.e. if (e1,e2) in dict, then (e2,e1) in dict 
    inv_edges={}
    for e in edges.keys():
        e1,e2=e
        x=edges[e]
        inv_edges[(e2,e1)]=x
    edges={**edges, **inv_edges}

    return (edges,properties)


def increase_modulo(x):
    return (x+1)%2

def sweep_hadamard(circuit, num_qubit):
    h_list=[0]*num_qubit
    new_circuit=[]
    for l in range(len(circuit)):
        if circuit[l][0]=='H':
            qubit=circuit[l][1]
            h_list[qubit]=increase_modulo(h_list[qubit])
        elif (circuit[l][0]=='x-spider') or (circuit[l][0]=='z-spider'):
            qubit=circuit[l][2]
            if h_list[qubit]==1:
                new_circuit.append(('H', qubit))
                h_list[qubit]=0
            new_circuit.append(circuit[l])
        elif circuit[l][0]=='CX':
            qubit1, qubit2=circuit[l][1]
            if h_list[qubit1]==1:
                new_circuit.append(('H', qubit1))
                h_list[qubit1]=0
            if h_list[qubit2]==1:
                new_circuit.append(('H', qubit2))
                h_list[qubit2]=0
            new_circuit.append(circuit[l])
        else:
            assert 0, 'not implemented error'
    # add Hadamard gates left in list:
    for k in range(num_qubit):
        if h_list[k]==1:
            new_circuit.append(('H', k))
    return new_circuit


def get_permutations(list1, list2):
    '''Created by ChatGPT'''
    swap_ops = []
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            j = list1.index(list2[i])
            list1[i], list1[j] = list1[j], list1[i]
            swap_ops.append((i, j))
    return swap_ops
