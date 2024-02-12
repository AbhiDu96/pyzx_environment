import gymnasium as gym
import numpy as np
import pyzx as zx
import qiskit
#from pyzx import rules
import torch
import re
from qiskit import QuantumCircuit, execute
from qiskit import qasm2
from qiskit import Aer
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, average_gate_fidelity
import torch_geometric.transforms as T
import copy
from fractions import Fraction
from .rules import custom_rules_chrisitan as rules
#import custom_rules_pyzx as rules
from .circuit_utils.circuit_generator import random_circuit
from .circuit_utils.reward_calculator import tcount_from_graph, check_equality

# For profilining
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile


import logging
import logging
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)


from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")



from .circuit_utils.grpah_format_converter_indexAdjusted import pyzx_to_heterogeneous_torchData, pyzy_to_homogeneous_torchData
from .circuit_utils.circuit_extractor import Circuit_extractor




class pyZX_env(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_
    """
    def __init__(self, n_qubits = 5, depth = 250, rules_list = None, max_steps=100, h_ratio = 0.3, t_ratio = 0.5, mq_ratio = 0.1,
        graph_type = "homogeneous", random_location=True, add_no_action=False) -> None:
        super().__init__()

        self.h_ratio = h_ratio
        self.t_ratio = t_ratio
        self.mq_ratio = mq_ratio
        self.n_qubits = n_qubits
        self.n_depth = depth
        self.graph_type = graph_type
        self.add_no_action = add_no_action
        if graph_type == "homogeneous":
            self.converter = pyzy_to_homogeneous_torchData
        else:
            self.converter = pyzx_to_heterogeneous_torchData
        self.random_location=random_location
        
        if rules_list != None:
            self.rules_list = rules_list
        else:
            self.rule_func_list = dir(rules)
            self.rules_list = [r for r in self.rule_func_list if "match_" in r]

        self.state_zx_graph_initital = None
        self.state_qiskit_circuit_initial = None
        self.state_zx_graph = None
        self.state = None

        self.action_space = gym.spaces.Discrete(len(self.rules_list))
        self._max_episode_steps = max_steps
        self.step_counter = 0
        # this is just a dummy to make gymnasium happy
        self.observation_space = gym.spaces.Discrete(5)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        
        self.step_counter = 0
        circuit_generated = False
        while not circuit_generated:
            init_success = False
            while not init_success:
                #circuit = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.n_qubits,depth=self.n_depth,clifford=False, p_had=self.h_ratio, p_t=self.t_ratio)
                circuit = random_circuit(n_qubit=self.n_qubits, num_gates=self.n_depth, p_two_qubit=self.mq_ratio, p_H=self.h_ratio, 
                        p_z=self.t_ratio, p_x=1-(self.mq_ratio+self.h_ratio+self.t_ratio), clifford_plus_T=True)
                
                # extract circut
                #initital_circuit_graph = zx.Circuit.from_qasm(self.state_qiskit_circuit_initial.qasm()).to_graph()
                initital_circuit_graph = circuit.to_graph()
                initital_circuit_graph_clone = initital_circuit_graph.clone()
                init_circuit_extractor = Circuit_extractor(initital_circuit_graph_clone)
                init_success = init_circuit_extractor.extract_circuit()
            
            self.reduced_zx_graph = initital_circuit_graph.clone()
            zx.full_reduce(self.reduced_zx_graph)
            
            self.baseline_reward = tcount_from_graph(self.reduced_zx_graph)
            full_t_count = tcount_from_graph(initital_circuit_graph)
            if full_t_count >= 10:
                circuit_generated = True




        self.state_qiskit_circuit_initial = qiskit.QuantumCircuit.from_qasm_str(init_circuit_extractor.zx_circuit.to_qasm())
        self.state_qiskit_circuit_initial = qiskit.transpile(self.state_qiskit_circuit_initial, optimization_level=3, basis_gates=['h', 'cx', 'rz'])
        initial_circuit = zx.Circuit.from_graph(initital_circuit_graph, split_phases=True)
        
        self.qiskit_reduced_zx = qiskit.QuantumCircuit.from_qasm_str(zx.extract_circuit(self.reduced_zx_graph.clone()).to_qasm())
        self.initial_gate_count = initial_circuit.stats_dict()['twoqubit']   #np.sum(list(self.state_qiskit_circuit_initial.count_ops().values()))
        self.full_reduced_gate_count = zx.extract_circuit(self.reduced_zx_graph.clone()).stats_dict()['twoqubit'] #np.sum(list(self.qiskit_reduced_zx.count_ops().values()))
            
        self.state_zx_graph_initital = initital_circuit_graph.clone()
        self.state_zx_graph = initital_circuit_graph.clone()
        
        
        #check_eq = check_equality(self.reduced_zx_graph, self.state_zx_graph_initital)

        self.state = self.converter(self.state_zx_graph)
        self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
        self.state = T.ToUndirected()(self.state)
        self.compute_action_masks()
        return ([self.state, self.action_masks, self.state_zx_graph, self.node_masks, self.edge_masks, self.rule_mask], {})
    
    def compute_action_masks(self):
        self.action_masks = []
        self.node_masks = []
        self.edge_masks = []
        self.rule_mask = []
        for r in self.rules_list:
            not_possible = np.zeros(1)
            rule_flag = 0
            #node_mask = np.zeros(self.state_zx_graph._vindex, dtype=int)
            #edge_mask = np.zeros(2*self.state_zx_graph.nedges, dtype=int)
            node_mask = np.zeros(len(self.node_index_mapping), dtype=int)
            edge_mask = np.zeros(2*self.state_zx_graph.nedges, dtype=int)
            match = getattr(rules, r)
            match_tupples = match(self.state_zx_graph)
            if len(match_tupples) > 0:
                rule_flag = 1
                if type(match_tupples[0]) == tuple:
                    #match_tupples_list = [*match_tupples]
                    match_tupples_list = []
                    for element in match_tupples:
                        #check = np.where(np.array(list(self.state_zx_graph.ty.keys())) == element[0])[0]
                        match_tupples_list.append((int(np.where(self.node_index_mapping == element[0])[0]), int(np.where(self.node_index_mapping == element[1])[0])))
                    values, indices = torch.topk(((self.state.edge_index == torch.Tensor(match_tupples_list).unsqueeze(-1)).all(dim=1)).int(), 1, 1)
                    indices = indices[values!=0]
                    edge_mask[indices.numpy()] = 1
                else:
                    match_tupples = np.searchsorted(self.node_index_mapping, match_tupples)
                    node_mask[match_tupples] = 1
            else:
                not_possible[0] = 1
            mask = np.concatenate([not_possible, node_mask, edge_mask])
            self.node_masks.append(node_mask)
            self.edge_masks.append(edge_mask)
            self.rule_mask.append(rule_flag)
            self.action_masks.append(mask)
        if self.add_no_action:
            not_possible = np.ones(1)
            node_mask = np.zeros(self.state_zx_graph._vindex, dtype=int)
            edge_mask = np.zeros(2*self.state_zx_graph.nedges, dtype=int)
            mask = np.concatenate([not_possible, node_mask, edge_mask])
            self.node_masks.append(node_mask)
            self.edge_masks.append(edge_mask)
            self.rule_mask.append(1)
            self.action_masks.append(mask)
        self.edge_masks = np.array(self.edge_masks, dtype=np.float32)
        self.node_masks = np.array(self.node_masks, dtype=np.float32)
        self.rule_mask = np.array(self.rule_mask, dtype=np.float32)
        self.action_masks = np.array(self.action_masks, dtype=np.float32)


    def select_match_tupples(self, action):

        match_str = self.rules_list[action]
        match_name = re.search('match_(.*)', match_str)
        if match_name == None:
            match_name = re.search('match_(.*)_', match_str)
        match_name = match_name.group(1)        
        match = getattr(rules, match_str)
        match_tupples = match(self.state_zx_graph)
        match_num = 0
        if len(match_tupples) > 0:
            match_num = np.random.randint(0, len(match_tupples))
        
        return match_name, match_tupples, match_num

    @profile
    def step(self, action: ActType, position: int | None = None, location: int | None = None) -> Tuple[ObsType, float, bool, bool, dict]:
        #print("entry step", len(list(self.state_zx_graph.ty.keys())))
        #err_msg = f"{action!r} ({type(action)}) invalid"
        #assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        truncated = False
        terminated = False

        """if action == self.action_space.n:
            done = True"""
        info = {}
        info["applied_rule"] = None
        info["match_tupples"] = None
        info["match_num"] = None
        info["init_circuit"] = None
        info["final_circuit"] = None
        info["depth_reduction"] = None
        info["gate_count_reduction"] = None
        reward = 0

        if action != self.action_space.n:
            
            if self.action_masks[action][0] != 1:
                if not terminated :
                    match_name, match_tupples, match_num = self.select_match_tupples(action)
                    
                    rewrite = getattr(rules, match_name)
                    info["applied_rule"] = match_name
                    info["match_tupples"] = match_tupples
                    if location is None:
                        if type(match_tupples[0]) == tuple:
                            relative_position = position - self.state.num_nodes
                            location = self.state.edge_index[:, relative_position]
                            location = (location[0].item(), location[1].item())
                            lookup = list(self.state_zx_graph.ty.keys())
                            #print("location indices raw",location,"graph state",self.state,"zx graph size",len(lookup))
                            location = (lookup[location[0]], lookup[location[1]])
                        else:
                            location = position
                            #l = len(list(self.state_zx_graph.ty.keys()))-1
                            #print("position length", l, "location",location,self.state)
                            location = list(self.state_zx_graph.ty.keys())[location]
                    
                    #input_output = self.state_zx_graph.inputs() + self.state_zx_graph.outputs()

                    if len(match_tupples) > 0:
                        prev_graph = self.state_zx_graph.clone()

                        try:
                            if match_name == "unspider":
                                info["match_num"] = position
                                neighbor=[list(self.state_zx_graph.neighbors(location))[0]]
                                new_phase=Fraction(1,1)
                                rules.unspider(self.state_zx_graph, [location,neighbor, new_phase])
                            else:
                                info["match_num"] = position
                                rules.apply_rule(g=self.state_zx_graph, rewrite=rewrite, m=location)

                            temp = self.state_zx_graph.clone()
                            zx.full_reduce(temp)
                            #current_t_count = tcount_from_graph(self.state_zx_graph.clone())
                            current_t_count = tcount_from_graph(temp)
                            current_gate_count = zx.extract_circuit(temp).stats_dict()['twoqubit']
                            reward = 2*(self.baseline_reward - current_t_count) + (self.initial_gate_count  - current_gate_count)
                        
                        except Exception as e:
                            logger.exception(e)
                            reward = -99
                            self.state_zx_graph = prev_graph
                        self.state = self.converter(self.state_zx_graph)
                        self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
                        self.state = T.ToUndirected()(self.state)
                        self.compute_action_masks()
            else:
                print("ACTION NOT MASKED!!!")
                reward = -199
            
        else:
            info["applied_rule"] = "No rule applied"

        self.step_counter += 1
            
        if self.step_counter >= self._max_episode_steps:
            truncated = True
        terminated =False
        
        if terminated or truncated:

            #extrator = Circuit_extractor(self.state_zx_graph)
            #success = extrator.extract_circuit()
            zx.full_reduce(self.state_zx_graph)
            success = check_equality(self.state_zx_graph_initital, self.state_zx_graph)
            

            if success:
                print('Unitary check passed')
                current_t_count = tcount_from_graph(self.state_zx_graph)
                current_gate_count = zx.extract_circuit(self.state_zx_graph.clone()).stats_dict()['twoqubit']
                reward = 2*(self.baseline_reward - current_t_count) + (self.initial_gate_count  - current_gate_count)
                q_circ_og = qiskit.QuantumCircuit.from_qasm_str(zx.extract_circuit(self.state_zx_graph.clone()).to_qasm())


                info["init_circuit"] = self.state_qiskit_circuit_initial
                info["init_circuit_gate_count"] = self.initial_gate_count
                info["init_circuit_t_count"] = tcount_from_graph(self.state_zx_graph_initital)


                info["final_circuit"] = q_circ_og
                info["final_circuit_gate_count"] = current_gate_count
                info["final_circuit_t_count"] = tcount_from_graph(self.state_zx_graph)

                info["gate_count_reduction"]  = self.initial_gate_count  - current_gate_count
                info["full_reduce_circuit"]  = self.qiskit_reduced_zx
                info["full_reduce_gate_count"]  = self.full_reduced_gate_count
                info["full_reduce_circuit_t_count"] = tcount_from_graph(self.reduced_zx_graph)
                
                #print("done triggered: ", reward)

            else:
                reward = -1000

            if reward == 0:
                reward = -2    
        # print("exit step", len(list(self.state_zx_graph.ty.keys())))
        self.state = self.converter(self.state_zx_graph)
        self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
        self.state = T.ToUndirected()(self.state)
        self.compute_action_masks()

        return [self.state, self.action_masks, self.state_zx_graph, self.node_masks, self.edge_masks, self.rule_mask], reward, terminated, truncated, info
        
