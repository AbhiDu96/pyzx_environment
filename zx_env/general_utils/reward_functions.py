import pyzx as zx
from zx_env.circuit_utils.circuit_extractor import extract_circuit
from zx_env.general_utils.utils import tcount_from_graph


def absolute_t_count_reward(zx_graph, baseline_t_count, baseline_cnot_count, pyzx_t_count=None, pyzx_cnot_count=None, circuit_extract_method="custom"):
    return baseline_t_count - tcount_from_graph(zx_graph)

def absolute_cnot_count_reward(zx_graph, baseline_t_count, baseline_cnot_count, pyzx_t_count=None, pyzx_cnot_count=None, circuit_extract_method="custom"):
    level = 5
    if circuit_extract_method == "custom":
        (circuit, level) = extract_circuit(zx_graph)
    else:
        zx.full_reduce(zx_graph)
        circuit = zx.extract_circuit(zx_graph)
    
    current_cnot_count = circuit.stats_dict()['twoqubit']

    return baseline_cnot_count - current_cnot_count, level

def normalized_t_count_reward(zx_graph, baseline_t_count, baseline_cnot_count, pyzx_t_count=None, pyzx_cnot_count=None, circuit_extract_method="custom"):
    return 1- (tcount_from_graph(zx_graph)/baseline_t_count)

def normalized_cnot_count_reward(zx_graph, baseline_t_count, baseline_cnot_count, pyzx_t_count=None, pyzx_cnot_count=None, circuit_extract_method="custom"):
    level = 5
    if circuit_extract_method == "custom":
         (circuit, level) = extract_circuit(zx_graph)
    else:
        zx.full_reduce(zx_graph)
        circuit = zx.extract_circuit(zx_graph)
    
    current_cnot_count = circuit.stats_dict()['twoqubit']

    return 1 - (current_cnot_count/(max(baseline_cnot_count,1e-5))), level

def pyzx_normalized_t_count_reward(zx_graph, baseline_t_count, baseline_cnot_count, pyzx_t_count=None, pyzx_cnot_count=None, circuit_extract_method="custom"):

    current_t_count = tcount_from_graph(zx_graph)
    return 1 - ((baseline_t_count+current_t_count)/(baseline_t_count+pyzx_t_count))


def pyzx_normalized_cnot_count_reward(zx_graph, baseline_t_count, baseline_cnot_count, pyzx_t_count=None, pyzx_cnot_count=None, circuit_extract_method="custom"):
    level=5
    if circuit_extract_method == "custom":
         (circuit, level) = extract_circuit(zx_graph)
    else:
        zx.full_reduce(zx_graph)
        circuit = zx.extract_circuit(zx_graph)
    
    current_cnot_count = circuit.stats_dict()['twoqubit']

    return 1 - ((baseline_cnot_count+current_cnot_count)/(baseline_cnot_count+pyzx_cnot_count)), level