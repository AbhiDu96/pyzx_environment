import gymnasium as gym
import numpy as np
import pyzx as zx
import torch
import re
import torch_geometric.transforms as T
import copy
from fractions import Fraction
from zx_env.rules import custom_rules as rules
from zx_env.circuit_utils.circuit_generator import random_circuit
import zx_env.general_utils.reward_functions as rf
from zx_env.general_utils.utils import check_equality, tcount_from_graph
from zx_env.circuit_utils.circuit_extractor import extract_circuit
from zx_env.circuit_utils.grpah_format_converter_indexAdjusted import pyzx_to_heterogeneous_torchData, pyzy_to_homogeneous_torchData
import glob
# For profilining
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

from matplotlib import pyplot as plt

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



class zx_env(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_
    """
    def __init__(self, n_qubits = 5, depth = 250, rules_list = None, max_steps=100, h_ratio = 0.3, t_ratio = 0.5, mq_ratio = 0.1,
        graph_type = "homogeneous", random_location=True, add_no_action=False,
        mutate_graph=True, mutate_probability = 0.5, mutation_steps=100, min_t_count_diff=0.1,
        reward_fn="normalized_t_count_reward", circuit_extraction_type="custom", negative_reward_mean=-0.1, negative_reward_std=0.0, full_fuse_every_step=False,reduce_at_reset=False) -> None:
        super().__init__()

        self.h_ratio = h_ratio
        self.t_ratio = t_ratio
        self.mq_ratio = mq_ratio
        self.n_qubits = n_qubits
        self.n_depth = depth
        self.graph_type = graph_type
        self.add_no_action = add_no_action
        self.mutate_graph = mutate_graph
        self.mutate_probability = mutate_probability
        self.mutation_steps = mutation_steps
        self.min_t_count_diff = min_t_count_diff
        self.circuit_extraction_type = circuit_extraction_type
        self.negative_reward_mean = negative_reward_mean
        self.negative_reward_std = negative_reward_std
        if reward_fn == "normalized_t_count_reward":
            self.reward_fn = rf.normalized_t_count_reward
        elif reward_fn == "absolute_t_count_reward":
            self.reward_fn = rf.absolute_t_count_reward
            #self.negative_reward = -2
        elif reward_fn == "absolute_cnot_count_reward":
            self.reward_fn = rf.absolute_cnot_count_reward
            #self.negative_reward = -2
        elif reward_fn == "normalized_cnot_count_reward":
            self.reward_fn = rf.normalized_cnot_count_reward
        elif reward_fn == "pyzx_normalized_t_count_reward":
            self.reward_fn = rf.pyzx_normalized_t_count_reward
        elif reward_fn == "pyzx_normalized_cnot_count_reward":
            self.reward_fn = rf.pyzx_normalized_cnot_count_reward
        else:
            self.reward_fn = reward_fn


        if graph_type == "homogeneous":
            self.converter = pyzy_to_homogeneous_torchData
        else:
            self.converter = pyzx_to_heterogeneous_torchData
        self.random_location=random_location
        
        if rules_list is not None:
            #self.rule_func_list = dir(rules)
            self.rules_list = ["match_"+r for r in rules_list]
        else:
            self.rule_func_list = dir(rules)
            self.rules_list = [r for r in self.rule_func_list if "match_" in r]
        print("rules",self.rules_list)
        self.state_zx_graph_initital = None
        self.state_zx_graph = None
        self.state = None

        self.action_space = gym.spaces.Discrete(len(self.rules_list))
        self._max_episode_steps = max_steps
        self.step_counter = 0
        # this is just a dummy to make gymnasium happy
        self.observation_space = gym.spaces.Discrete(5)
        self.full_fuse_every_step=full_fuse_every_step
        self.reduce_at_reset = reduce_at_reset
        self.bench_mark()


    def sample_circuit(self):
        n_qubits = self.n_qubits if isinstance(self.n_qubits, int) else np.random.randint(*self.n_qubits)
        n_depth = self.n_depth if isinstance(self.n_depth, int) else np.random.randint(*self.n_depth)
        mq_ratio= self.mq_ratio if isinstance(self.mq_ratio, float) else np.random.uniform(*self.mq_ratio)
        h_ratio= self.h_ratio if isinstance(self.h_ratio, float) else np.random.uniform(*self.h_ratio)
        t_ratio = self.t_ratio if isinstance(self.t_ratio, float) else np.random.uniform(*self.t_ratio)
        p_x = 1-(mq_ratio+h_ratio+(t_ratio/2))
        circuit = random_circuit(n_qubit=n_qubits, num_gates=n_depth, p_two_qubit=mq_ratio, p_H=h_ratio, 
            p_z=t_ratio/2, p_x=p_x, clifford_plus_T=True)
        return circuit

    def bench_mark(self, path="zx_env/bench_mark_circuits/"):

        self.benchmakr_circuit_name = glob.glob(path+"*/*", recursive=True)
        self.benchmark_graphs = []
        for circ in self.benchmakr_circuit_name:
            self.benchmark_graphs.append(zx.Circuit.from_quipper_file(circ))
        
    def mk_features(self):
        circ = extract_circuit(self.state_zx_graph)[0]
        stats = circ.stats_dict(depth=True)
        gates = stats["gates"]
        tcount = stats["tcount"]
        clifford = stats["clifford"]
        twoqubit = stats["twoqubit"]
        cnot = stats["cnot"]
        had = stats["had"]
        depth = stats["depth"]
        depth_cz = stats["depth_cz"]
        edges = self.state_zx_graph.num_edges()
        feat = torch.tensor([gates,tcount,clifford,twoqubit,cnot,had,depth,depth_cz,edges]).float()
        return feat

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, initital_circuit_graph = None,
                simplfy_initial_circuit = False) -> Tuple[ObsType, dict]:
        
        self.step_counter = 0
        circuit_generated = False
        #simplfy_initial_circuit = simplfy_initial_circuit or self.reduce_at_reset
        print("reset start")

        if initital_circuit_graph == None:
            while not circuit_generated:
                circuit = self.sample_circuit()
                initital_circuit_graph = circuit.to_graph()
                
                if simplfy_initial_circuit:
                    # single extract pass through to reduce simple cancelations 
                    (initial_circuit, _) = extract_circuit(initital_circuit_graph)
                    initital_circuit_graph = initial_circuit.to_graph()
                else:
                    initial_circuit = copy.deepcopy(circuit)
                self.baseline_t_count = tcount_from_graph(initital_circuit_graph)
                self.reduced_zx_graph = initital_circuit_graph.clone()
                zx.full_reduce(self.reduced_zx_graph)

                self.pyzx_t_count = tcount_from_graph(self.reduced_zx_graph)
                if np.random.random() < 0.5:
                    if (self.baseline_t_count-self.pyzx_t_count)/(self.baseline_t_count+1e-5) >= self.min_t_count_diff:
                        circuit_generated = True
                else:
                    if self.baseline_t_count >= 10:
                        circuit_generated = True        
        else:
            if simplfy_initial_circuit:
                (initial_circuit, _) = extract_circuit(initital_circuit_graph)
                initital_circuit_graph = initial_circuit.to_graph()
            else:
                initial_circuit = zx.Circuit.from_graph(initital_circuit_graph)
            self.baseline_t_count = tcount_from_graph(initital_circuit_graph)
            self.reduced_zx_graph = initital_circuit_graph.clone()
            zx.full_reduce(self.reduced_zx_graph)
            self.pyzx_t_count = tcount_from_graph(self.reduced_zx_graph)
        self.state_circuit_initial = initial_circuit
        self.state_zx_graph_initital = initital_circuit_graph.clone()
        self.state_zx_graph = initital_circuit_graph.clone()

        # extract pyzx based graph and circuit
        self.reduced_zx_circuit = zx.extract_circuit(self.reduced_zx_graph)

        self.baseline_cnot_count = initial_circuit.stats_dict()['twoqubit']
        self.pyzx_cnot_count = self.reduced_zx_circuit.stats_dict()['twoqubit']
        print("pre morph")
        # morph the graph
        if np.random.rand() < self.mutate_probability:
            mutation_counter = 0
            for mutation_counter in range(self.mutation_steps):
                action = np.random.randint(len(self.rules_list))
                match_name, match_tupples, match_num = self.select_match_tupples(action)
                rewrite = getattr(rules, match_name)
                if len(match_tupples)>0:
                    if match_name == "unspider":
                        neighbor=[list(self.state_zx_graph.neighbors(match_tupples[match_num]))[0]]
                        new_phase=Fraction(1,1)
                        rules.unspider(self.state_zx_graph, [match_tupples[match_num],neighbor, new_phase])
                    else:
                        rules.apply_rule(g=self.state_zx_graph, rewrite=rewrite, m=match_tupples[match_num])
                    #mutation_counter += 1
        print("post morph")
        if self.reduce_at_reset:
            print("reform")
            rules.full_fuse(self.state_zx_graph)


        self.state = self.converter(self.state_zx_graph)
        self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
        self.state = T.ToUndirected()(self.state)
        self.compute_action_masks()
        #fig=zx.draw_matplotlib(self.state_zx_graph)
        #fig.savefig("test.png")
        print("N actions",self.action_masks[:,1:].sum())
        info = dict()
        reward, info["level"] = self.reward_fn(zx_graph=self.state_zx_graph.clone(), baseline_t_count=self.baseline_t_count, baseline_cnot_count=self.baseline_cnot_count,
                                pyzx_t_count=self.pyzx_t_count, pyzx_cnot_count=self.pyzx_cnot_count, circuit_extract_method=self.circuit_extraction_type)
        info["reward"]=reward
        info["feats"]=self.mk_features()
        return ([self.state, self.action_masks, self.state_zx_graph, self.node_masks, self.edge_masks, self.rule_mask], info)
    
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


    #@profile
    def step(self, action: ActType, position: int | None = None, location: int | None = None,pyzx_state : Any | None = None) -> Tuple[ObsType, float, bool, bool, dict]:
        #print("entry step", len(list(self.state_zx_graph.ty.keys())))
        #err_msg = f"{action!r} ({type(action)}) invalid"
        #assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        if pyzx_state is not None:
            self.state_zx_graph = pyzx_state.clone()
            self.state = self.converter(self.state_zx_graph)
            self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
            self.state = T.ToUndirected()(self.state)
            self.compute_action_masks()
            #assert self.action_masks[:,1:][action,position]==1


        truncated = False
        terminated = False
        isdead=False

        """if action == self.action_space.n:
            done = True"""
        info = {}
        info["applied_rule"] = None
        info["match_tupples"] = None
        info["match_num"] = None
        info["init_circuit"] = None
        info["final_circuit"] = None
        info["depth_reduction"] = None
        info["cnot_count_reduction"] = None
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
                            location = list(self.state_zx_graph.ty.keys())[location]

                    if len(match_tupples) > 0:
                        prev_graph = self.state_zx_graph.clone()

                        try:
                            #print("executing action",match_name,"at position",position)
                            if match_name == "unspider":
                                info["match_num"] = position
                                neighbor=[list(self.state_zx_graph.neighbors(location))[0]]
                                new_phase=Fraction(1,1)
                                rules.unspider(self.state_zx_graph, [location,neighbor, new_phase])
                            else:
                                info["match_num"] = position
                                rules.apply_rule(g=self.state_zx_graph, rewrite=rewrite, m=location)
                            #print("applied action")
                            reward, info["level"] = self.reward_fn(zx_graph=self.state_zx_graph.clone(), baseline_t_count=self.baseline_t_count, baseline_cnot_count=self.baseline_cnot_count,
                                pyzx_t_count=self.pyzx_t_count, pyzx_cnot_count=self.pyzx_cnot_count, circuit_extract_method=self.circuit_extraction_type)
                        
                        except Exception as e:
                            #print("EXCEPTION DURING EXTRACTION")
                            logger.exception(e)
                            reward = -10
                            self.state_zx_graph = prev_graph
                            terminated=True
                            isdead=True

            else:
                print("ACTION NOT MASKED!!!")
                reward = -199            
        else:
            info["applied_rule"] = "No rule applied"
        if self.full_fuse_every_step:
            rules.full_fuse(self.state_zx_graph)
        self.step_counter += 1
            
        if self.step_counter >= self._max_episode_steps:
            truncated = True
        self.state = self.converter(self.state_zx_graph)
        self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
        self.state = T.ToUndirected()(self.state)
        self.compute_action_masks()
        if np.all(self.action_masks[:,1:]==0):
            print("no action possible")
            terminated = True
        if (terminated or truncated) and not isdead:
            print("TERMINATED",terminated,truncated,isdead)
            
            success = check_equality(self.state_zx_graph_initital.clone(), self.state_zx_graph.clone())
            

            if success:
                reward, info["level"] = self.reward_fn(zx_graph=self.state_zx_graph.clone(), baseline_t_count=self.baseline_t_count, baseline_cnot_count=self.baseline_cnot_count,
                                pyzx_t_count=self.pyzx_t_count, pyzx_cnot_count=self.pyzx_cnot_count, circuit_extract_method=self.circuit_extraction_type)

                current_t_count = tcount_from_graph(self.state_zx_graph)

                if self.circuit_extraction_type == "custom":
                    (circuit, _) = extract_circuit(self.state_zx_graph.clone())
                else:
                    zx.full_reduce(self.state_zx_graph.clone())
                    circuit = zx.extract_circuit(self.state_zx_graph.clone())

                current_cnot_count = circuit.stats_dict()['twoqubit']
                


                info["init_circuit"] = self.state_circuit_initial
                info["init_circuit_cnot_count"] = self.baseline_cnot_count
                info["init_circuit_t_count"] = self.baseline_t_count


                info["final_circuit"] = circuit
                info["final_circuit_cnot_count"] = current_cnot_count
                info["final_circuit_t_count"] = current_t_count

                info["full_reduce_circuit"]  = self.reduced_zx_circuit
                info["full_reduce_cnot_count"]  = self.pyzx_cnot_count
                info["full_reduce_circuit_t_count"] = self.pyzx_t_count
                

            else:
                reward = -1000

            if reward == 0:
                reward = self.negative_reward_mean + self.negative_reward_std*np.random.randn()  
        
        
        
        info["feats"]=self.mk_features()
        return [self.state, self.action_masks, self.state_zx_graph, self.node_masks, self.edge_masks, self.rule_mask], reward, terminated, truncated, info
        
