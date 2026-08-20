import gymnasium as gym
from collections.abc import Sequence
import numpy as np
import pyzx as zx
import torch
import re
import torch_geometric.transforms as T
import copy
from dataclasses import dataclass
from fractions import Fraction
from zx_env.rules import custom_rules as rules
from zx_env.circuit_utils.circuit_generator import random_circuit
import zx_env.general_utils.reward_functions as rf
from zx_env.general_utils.utils import check_equality, tcount_from_graph
from zx_env.circuit_utils.circuit_extractor import extract_circuit
from zx_env.circuit_utils.graph_format_converter_index_adjusted import pyzx_to_heterogeneous_torchData, pyzx_to_homogeneous_torchData
import glob

# For profiling: provide a pass-through @profile when line_profiler is not active
import builtins

try:
    builtins.profile
except AttributeError:
    def profile(func): return func
    builtins.profile = profile

import logging
logging.basicConfig(filename='error.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


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


# Reward values used for terminal / failure conditions
REWARD_REWRITE_EXCEPTION = -10    # a rewrite raised an exception while being applied
REWARD_ILLEGAL_ACTION = -199      # a masked-out (not applicable) action was selected
REWARD_NOT_EQUIVALENT = -1000     # final graph is not equivalent to the original circuit
REWARD_FN_MAP = {
    "normalized_t_count_reward": rf.normalized_t_count_reward,
    "absolute_t_count_reward": rf.absolute_t_count_reward,
    "absolute_cnot_count_reward": rf.absolute_cnot_count_reward,
    "normalized_cnot_count_reward": rf.normalized_cnot_count_reward,
    "pyzx_normalized_t_count_reward": rf.pyzx_normalized_t_count_reward,
    "pyzx_normalized_cnot_count_reward": rf.pyzx_normalized_cnot_count_reward,
}


@dataclass(frozen=True)
class ZXEnvConfig:
    n_qubits: Any = 5
    depth: Any = 250
    rules_list: list[str] | None = None
    max_steps: int = 100
    h_ratio: Any = 0.3
    t_ratio: Any = 0.5
    mq_ratio: Any = 0.1
    graph_type: str = "homogeneous"
    random_location: bool = True
    add_no_action: bool = False
    mutate_graph: bool = True
    mutate_probability: float = 0.5
    mutation_steps: int = 100
    min_t_count_diff: float = 0.1
    reward_fn: Any = "normalized_t_count_reward"
    circuit_extraction_type: str = "custom"
    negative_reward_mean: float = -0.1
    negative_reward_std: float = 0.0
    full_fuse_every_step: bool = False
    reduce_at_reset: bool = False


class ZXEnv(gym.Env):
    """Gymnasium environment for optimizing quantum circuits via ZX-diagram rewriting.

    The environment samples a random Clifford+T circuit, converts it to a ZX graph, and
    exposes the graph (plus action masks and a feature vector) as an observation. Actions
    correspond to PyZX graph-rewriting rules applied at a chosen node or edge. The goal is
    to reduce the two-qubit (CNOT/CZ) and/or T gate count of the extracted circuit while
    keeping it equivalent to the original.
    """
    _FEATURE_STAT_KEYS: tuple[str, ...] = (
        "gates",
        "tcount",
        "clifford",
        "twoqubit",
        "had",
        "depth",
        "depth_cz",
    )

    def __init__(self, n_qubits = 5, depth = 250, rules_list = None, max_steps=100, h_ratio = 0.3, t_ratio = 0.5, mq_ratio = 0.1,
        graph_type = "homogeneous", random_location=True, add_no_action=False,
        mutate_graph=True, mutate_probability = 0.5, mutation_steps=100, min_t_count_diff=0.1,
        reward_fn="normalized_t_count_reward", circuit_extraction_type="custom", negative_reward_mean=-0.1, negative_reward_std=0.0, full_fuse_every_step=False,reduce_at_reset=False) -> None:
        super().__init__()

        cfg = ZXEnvConfig(
            n_qubits=n_qubits,
            depth=depth,
            rules_list=rules_list,
            max_steps=max_steps,
            h_ratio=h_ratio,
            t_ratio=t_ratio,
            mq_ratio=mq_ratio,
            graph_type=graph_type,
            random_location=random_location,
            add_no_action=add_no_action,
            mutate_graph=mutate_graph,
            mutate_probability=mutate_probability,
            mutation_steps=mutation_steps,
            min_t_count_diff=min_t_count_diff,
            reward_fn=reward_fn,
            circuit_extraction_type=circuit_extraction_type,
            negative_reward_mean=negative_reward_mean,
            negative_reward_std=negative_reward_std,
            full_fuse_every_step=full_fuse_every_step,
            reduce_at_reset=reduce_at_reset,
        )

        self.h_ratio = cfg.h_ratio
        self.t_ratio = cfg.t_ratio
        self.mq_ratio = cfg.mq_ratio
        self.n_qubits = cfg.n_qubits
        self.n_depth = cfg.depth
        self.graph_type = cfg.graph_type
        self.add_no_action = cfg.add_no_action
        self.mutate_graph = cfg.mutate_graph
        self.mutate_probability = cfg.mutate_probability
        self.mutation_steps = cfg.mutation_steps
        self.min_t_count_diff = cfg.min_t_count_diff
        self.circuit_extraction_type = cfg.circuit_extraction_type
        self.negative_reward_mean = cfg.negative_reward_mean
        self.negative_reward_std = cfg.negative_reward_std
        self.reward_fn = self._resolve_reward_fn(cfg.reward_fn)

        self.converter = self._resolve_converter(cfg.graph_type)
        self.random_location = cfg.random_location

        self.rule_func_list, self.rules_list = self._resolve_rules_list(cfg.rules_list)
        logging.debug("rules: %s", self.rules_list)  # noqa: LOG015
        self.state_zx_graph_initial = None
        self.state_zx_graph = None
        self.state = None

        self.action_space = gym.spaces.Discrete(len(self.rules_list))
        self._max_episode_steps = cfg.max_steps
        self.step_counter = 0
        # this is just a dummy to make gymnasium happy
        self.observation_space = gym.spaces.Discrete(5)
        self.full_fuse_every_step = cfg.full_fuse_every_step
        self.reduce_at_reset = cfg.reduce_at_reset
        self.bench_mark()

    @classmethod
    def _resolve_reward_fn(cls, reward_fn: Any) -> Any:
        if isinstance(reward_fn, str):
            return REWARD_FN_MAP.get(reward_fn, reward_fn)
        return reward_fn

    @staticmethod
    def _resolve_converter(graph_type: str) -> Any:
        if graph_type == "homogeneous":
            return pyzx_to_homogeneous_torchData
        return pyzx_to_heterogeneous_torchData

    @staticmethod
    def _resolve_rules_list(rules_list: Sequence[str] | None) -> tuple[list[str], list[str]]:
        if rules_list is not None:
            return [], ["match_" + r for r in rules_list]
        rule_func_list = dir(rules)
        resolved_rules = [r for r in rule_func_list if "match_" in r]
        return rule_func_list, resolved_rules


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

    def bench_mark(self, path="zx_env/BenchmarkCircuits/"):

        self.benchmark_circuit_names = glob.glob(path+"*/*", recursive=True)
        self.benchmark_graphs = []
        for circ in self.benchmark_circuit_names:
            self.benchmark_graphs.append(zx.Circuit.from_quipper_file(circ))

    def compute_features(self):
        circ = extract_circuit(self.state_zx_graph)[0]
        stats = circ.stats_dict(depth=True)
        exp_qubits = self.n_qubits if isinstance(self.n_qubits, int) else np.mean(self.n_qubits)
        exp_gates = self.n_depth if isinstance(self.n_depth, int) else np.mean(self.n_depth)
        expected_size = exp_gates * exp_qubits
        feat_values = [stats[k] / expected_size for k in self._FEATURE_STAT_KEYS]
        feat_values.append(self.state_zx_graph.num_edges() / expected_size)
        feat = torch.tensor(feat_values).float()
        return feat

    def reset(self, *, seed: int | None = None, options: dict | None = None, initial_circuit_graph = None,
                simplify_initial_circuit = False) -> tuple[ObsType, dict]:

        self.step_counter = 0
        circuit_generated = False

        if initial_circuit_graph == None:
            while not circuit_generated:
                circuit = self.sample_circuit()
                initial_circuit_graph = circuit.to_graph()

                if simplify_initial_circuit:
                    # single extract pass through to reduce simple cancelations
                    (initial_circuit, _) = extract_circuit(initial_circuit_graph)
                    initial_circuit_graph = initial_circuit.to_graph()
                else:
                    initial_circuit = copy.deepcopy(circuit)
                self.baseline_t_count = tcount_from_graph(initial_circuit_graph)
                self.reduced_zx_graph = initial_circuit_graph.clone()
                zx.full_reduce(self.reduced_zx_graph)

                self.pyzx_t_count = tcount_from_graph(self.reduced_zx_graph)
                if np.random.random() < 0.5:
                    if (self.baseline_t_count-self.pyzx_t_count)/(self.baseline_t_count+1e-5) >= self.min_t_count_diff:
                        circuit_generated = True
                else:
                    if self.baseline_t_count >= 10:
                        circuit_generated = True
        else:
            logging.debug("loading pre-done circuit")  # noqa: LOG015
            if simplify_initial_circuit:
                (initial_circuit, _) = extract_circuit(initial_circuit_graph)
                initial_circuit_graph = initial_circuit.to_graph()
            else:
                initial_circuit = zx.Circuit.from_graph(initial_circuit_graph)
            self.baseline_t_count = tcount_from_graph(initial_circuit_graph)
            self.reduced_zx_graph = initial_circuit_graph.clone()
            zx.full_reduce(self.reduced_zx_graph)
            self.pyzx_t_count = tcount_from_graph(self.reduced_zx_graph)
        self.state_circuit_initial = initial_circuit
        self.state_zx_graph_initial = initial_circuit_graph.clone()
        self.state_zx_graph = initial_circuit_graph.clone()

        # extract pyzx based graph and circuit
        self.reduced_zx_circuit = zx.extract_circuit(self.reduced_zx_graph)

        self.baseline_cnot_count = initial_circuit.stats_dict()['twoqubit']
        self.pyzx_cnot_count = self.reduced_zx_circuit.stats_dict()['twoqubit']
        # morph the graph
        if np.random.rand() < self.mutate_probability:
            for mutation_counter in range(self.mutation_steps):
                action = np.random.randint(len(self.rules_list))
                match_name, match_tuples, match_num = self.select_match_tuples(action)
                rewrite = getattr(rules, match_name)
                if len(match_tuples)>0:
                    if match_name == "unspider":
                        neighbor=[next(iter(self.state_zx_graph.neighbors(match_tuples[match_num])))]
                        new_phase=Fraction(1,1)
                        rules.unspider(self.state_zx_graph, [match_tuples[match_num],neighbor, new_phase])
                    else:
                        rules.apply_rule(g=self.state_zx_graph, rewrite=rewrite, m=match_tuples[match_num])
        if self.reduce_at_reset:
            logging.debug("reform")  # noqa: LOG015
            rules.full_fuse(self.state_zx_graph)

        self.state = self.converter(self.state_zx_graph)
        self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
        self.state = T.ToUndirected()(self.state)
        self.compute_action_masks()
        info = {}
        reward, info["level"] = self.reward_fn(zx_graph=self.state_zx_graph.clone(), baseline_t_count=self.baseline_t_count, baseline_cnot_count=self.baseline_cnot_count,
                                pyzx_t_count=self.pyzx_t_count, pyzx_cnot_count=self.pyzx_cnot_count, circuit_extract_method=self.circuit_extraction_type)
        info["reward"]=reward
        info["feats"]=self.compute_features()
        info["action_mask"] = self.action_masks
        return ([self.state, self.action_masks, self.state_zx_graph, self.node_masks, self.edge_masks, self.rule_mask], info)

    def compute_action_masks(self):
        self.action_masks = []
        self.node_masks = []
        self.edge_masks = []
        self.rule_mask = []
        for r in self.rules_list:
            not_possible = np.zeros(1)
            rule_flag = 0
            node_mask = np.zeros(len(self.node_index_mapping), dtype=int)
            edge_mask = np.zeros(2*self.state_zx_graph.nedges, dtype=int)
            match = getattr(rules, r)
            match_tuples = match(self.state_zx_graph)
            if len(match_tuples) > 0:
                rule_flag = 1
                if type(match_tuples[0]) == tuple:
                    match_tuples_list = []
                    for element in match_tuples:
                        match_tuples_list.append((int(np.where(self.node_index_mapping == element[0])[0]), int(np.where(self.node_index_mapping == element[1])[0])))
                    values, indices = torch.topk(((self.state.edge_index == torch.Tensor(match_tuples_list).unsqueeze(-1)).all(dim=1)).int(), 1, 1)
                    indices = indices[values!=0]
                    edge_mask[indices.numpy()] = 1
                else:
                    match_tuples = np.searchsorted(self.node_index_mapping, match_tuples)
                    node_mask[match_tuples] = 1
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


    def select_match_tuples(self, action):

        match_str = self.rules_list[action]
        match_name = re.search('match_(.*)', match_str)
        if match_name == None:
            match_name = re.search('match_(.*)_', match_str)
        match_name = match_name.group(1)
        match = getattr(rules, match_str)
        match_tuples = match(self.state_zx_graph)
        match_num = 0
        if len(match_tuples) > 0:
            match_num = np.random.randint(0, len(match_tuples))

        return match_name, match_tuples, match_num


    def step(self, action: ActType, position: int | None = None, location: int | None = None, pyzx_state : Any | None = None) -> tuple[ObsType, float, bool, bool, dict]:
        assert self.state is not None, "Call reset before using step method."

        if pyzx_state is not None:
            self.state_zx_graph = pyzx_state.clone()
            self.state = self.converter(self.state_zx_graph)
            self.node_index_mapping = np.array(list(self.state_zx_graph.ty.keys()))
            self.state = T.ToUndirected()(self.state)
            self.compute_action_masks()

        truncated = False
        terminated = False
        isdead=False

        info = {}
        info["applied_rule"] = None
        info["match_tuples"] = None
        info["match_num"] = None
        info["init_circuit"] = None
        info["final_circuit"] = None
        info["depth_reduction"] = None
        info["cnot_count_reduction"] = None
        reward = 0

        if action != self.action_space.n:
            if self.action_masks[action][0] != 1:

                if not terminated :
                    match_name, match_tuples, _match_num = self.select_match_tuples(action)

                    rewrite = getattr(rules, match_name)
                    info["applied_rule"] = match_name
                    info["match_tuples"] = match_tuples
                    if location is None:
                        if type(match_tuples[0]) == tuple:
                            relative_position = position - self.state.num_nodes
                            location = self.state.edge_index[:, relative_position]
                            location = (location[0].item(), location[1].item())
                            lookup = list(self.state_zx_graph.ty.keys())
                            location = (lookup[location[0]], lookup[location[1]])
                        else:
                            location = position
                            location = list(self.state_zx_graph.ty.keys())[location]

                    if len(match_tuples) > 0:
                        prev_graph = self.state_zx_graph.clone()

                        try:
                            if match_name == "unspider":
                                info["match_num"] = position
                                neighbor=[next(iter(self.state_zx_graph.neighbors(location)))]
                                new_phase=Fraction(1,1)
                                rules.unspider(self.state_zx_graph, [location,neighbor, new_phase])
                            else:
                                info["match_num"] = position
                                rules.apply_rule(g=self.state_zx_graph, rewrite=rewrite, m=location)
                            reward, info["level"] = self.reward_fn(zx_graph=self.state_zx_graph.clone(), baseline_t_count=self.baseline_t_count, baseline_cnot_count=self.baseline_cnot_count,
                                pyzx_t_count=self.pyzx_t_count, pyzx_cnot_count=self.pyzx_cnot_count, circuit_extract_method=self.circuit_extraction_type)
                        except:
                            reward = REWARD_REWRITE_EXCEPTION
                            self.state_zx_graph = prev_graph
                            terminated=True
                            isdead=True

            else:
                logging.warning("ACTION NOT MASKED!!!")
                reward = REWARD_ILLEGAL_ACTION
                terminated=True
                truncated=True
                isdead=True
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
            logging.info("no action possible")
            terminated = True
        if (terminated or truncated) and not isdead:
            logging.info("terminated=%s truncated=%s isdead=%s", terminated, truncated, isdead)

            success = check_equality(self.state_zx_graph_initial.clone(), self.state_zx_graph.clone())


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
                
                # store the initial and final circuit information in the info dictionary
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
                reward = REWARD_NOT_EQUIVALENT

            if reward == 0:
                reward = self.negative_reward_mean + self.negative_reward_std*np.random.randn()



        info["feats"]=self.compute_features()
        info["action_mask"] = self.action_masks
        info["reward"]=reward
        return [self.state, self.action_masks, self.state_zx_graph, self.node_masks, self.edge_masks, self.rule_mask], reward, terminated, truncated, info
