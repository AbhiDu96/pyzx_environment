import pyzx as zx
import numpy as np
from torch_geometric.data import Data, HeteroData

#import scipy.sparse as sp
from torch_geometric.utils.undirected import is_undirected
import torch

def calculate_edge_COO_undirected(pyzx_graph):
    
    graph_dict = pyzx_graph.graph
    rows, cols, edge_features = [], [], []
    for k, values in graph_dict.items():
        for i, edge_type in values.items():
            rows.append(k)
            cols.append(i)
            edge_features.append(edge_type)
    #X = sp.csr_matrix((edge_features, (rows, cols)))
    #coo_mat = X.tocoo()
    coo_mat = np.vstack((rows, cols))
    return coo_mat, np.array(edge_features)

def calculate_edge_COO_directed(pyzx_graph):
    
    rows, cols, edge_features = [], [], []
    edge_Set = pyzx_graph.edge_set()

    for edge in edge_Set:
        rows.append(edge[0])
        cols.append(edge[1])
        edge_features.append(pyzx_graph.graph[edge[0]][edge[1]])

    coo_mat = np.vstack((rows, cols))
    return coo_mat, np.array(edge_features)


def pyzy_to_homogeneous_torchData(pyzx_graph, label=None, graph_type=None):
    
    if graph_type == None:
        graph_type = calculate_edge_COO_undirected
    
    #pyzx_graph = zx.Graph().from_json(g)
    coo_mat, edge_features = graph_type(pyzx_graph)
    
    node_phase = np.array([float(x) for x in list(pyzx_graph.phases().values())])*np.pi
    node_type = np.array(list(pyzx_graph.types().values()))
    keys = list(pyzx_graph.types())
    node_features = np.zeros((np.amax(keys)+1, 2))
    node_features[keys, :] = np.hstack((node_type.reshape(-1,1), node_phase.reshape(-1,1)))

    
    torch_data = Data(x=torch.tensor(node_features.astype(np.float32)), 
        edge_index=torch.tensor(coo_mat.astype(np.int64)), edge_attr=torch.tensor(edge_features.astype(np.float32)))
    
    return torch_data

def calculate_heterogeneous_edge_COO_directed(pyzx_graph, node_list1, node_list2):

    rows, cols, edge_features = [], [], []

    set_list = []

    #node_1_mapping = list(range(len(node_list1)))
    #node_2_mapping = list(range(len(node_list2)))

    for node_1 in node_list1:
        conected_nodes = pyzx_graph.graph[node_1]
        for node_2 in list(conected_nodes.keys()):
            if node_2 in node_list2 and set([node_1, node_2]) not in set_list:
    
                rows.append(int(np.where(node_list1 == node_1)[0]))
                cols.append(int(np.where(node_list2 == node_2)[0]))
                edge_features.append(conected_nodes[node_2])
                set_list.append(set([node_1, node_2]))
    
    coo_mat = np.vstack((rows, cols))
    return coo_mat, np.array(edge_features)

def calculate_heterogeneous_edge_COO_undirected(pyzx_graph, node_list1, node_list2):

    rows, cols, edge_features = [], [], []

    set_list = []

    for node_1 in node_list1:
        conected_nodes = pyzx_graph.graph[node_1]
        for node_2 in list(conected_nodes.keys()):
            if node_2 in node_list2 and set([node_1, node_2]) not in set_list:
                rows.append(node_1)
                cols.append(node_2)
                edge_features.append(conected_nodes[node_2])
                set_list.append(set([node_1, node_2]))

    coo_mat = np.vstack((rows+cols, cols+rows))
    edge_features = edge_features+edge_features
    return coo_mat, np.array(edge_features)

def pyzx_to_heterogeneous_torchData(g, label=None, graph_type=None):

    if graph_type == None:
        graph_type = calculate_heterogeneous_edge_COO_directed
    
    pyzx_graph = g

    #imp = zx.draw_matplotlib(pyzx_graph, labels=True, show_scalar=True)
    #imp.savefig("graph_2.png")

    input_indices = np.array(pyzx_graph.inputs())
    output_indices = np.array(pyzx_graph.outputs())
    x_spider_indices = np.where(np.array(list(pyzx_graph.types().values()), dtype=np.int64)==1)[0]
    z_spider_indices = np.where(np.array(list(pyzx_graph.types().values()), dtype=np.int64)==2)[0]

    current_node_list = np.array(list(pyzx_graph.graph.keys()))
    x_spider_indices = np.intersect1d(x_spider_indices, current_node_list)
    z_spider_indices = np.intersect1d(z_spider_indices, current_node_list)

    torch_data = HeteroData()

    torch_data['inNodes'].x = torch.tensor(np.ones((len(input_indices), 1), dtype=np.float32)*99)
    torch_data['outNodes'].x = torch.tensor(np.ones((len(output_indices), 1), dtype=np.float32)*(-99))
    torch_data['xSpiders'].x = torch.tensor(np.array([float(x) for x in list(pyzx_graph.phases().values())], dtype=np.float32)[x_spider_indices].reshape(-1,1)*np.pi)
    torch_data['zSpiders'].x = torch.tensor(np.array([float(x) for x in list(pyzx_graph.phases().values())], dtype=np.float32)[z_spider_indices].reshape(-1,1)*np.pi)

    # input connections
    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=input_indices, node_list2=x_spider_indices)
    torch_data['inNodes', 'inNodesTOxSpiders', 'xSpiders'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['inNodes', 'inNodesTOxSpiders', 'xSpiders'].edge_attr = torch.tensor(edge_features.astype(np.float32))

    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=input_indices, node_list2=z_spider_indices)
    torch_data['inNodes', 'inNodesTOzSpiders', 'zSpiders'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['inNodes', 'inNodesTOzSpiders', 'zSpiders'].edge_attr = torch.tensor(edge_features.astype(np.float32))



    # output connections
    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=x_spider_indices, node_list2=output_indices)
    torch_data['xSpiders', 'xSpidersTOoutNodes', 'outNodes'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['xSpiders', 'xSpidersTOoutNodes', 'outNodes'].edge_attr = torch.tensor(edge_features.astype(np.float32))

    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=z_spider_indices, node_list2=output_indices)
    torch_data['zSpiders', 'zSpidersTOoutNodes', 'outNodes'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['zSpiders', 'zSpidersTOoutNodes', 'outNodes'].edge_attr = torch.tensor(edge_features.astype(np.float32))

    # x to x connections

    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=x_spider_indices, node_list2=x_spider_indices)
    torch_data['xSpiders', 'xSpidersTOxSpiders', 'xSpiders'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['xSpiders', 'xSpidersTOxSpiders', 'xSpiders'].edge_attr = torch.tensor(edge_features.astype(np.float32))

    # z to z connections

    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=z_spider_indices, node_list2=z_spider_indices)
    torch_data['zSpiders', 'zSpidersTOzSpiders', 'zSpiders'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['zSpiders', 'zSpidersTOzSpiders', 'zSpiders'].edge_attr = torch.tensor(edge_features.astype(np.float32))

    # x to z connections

    coo_matrix, edge_features = graph_type(pyzx_graph=pyzx_graph, node_list1=x_spider_indices, node_list2=z_spider_indices)
    torch_data['xSpiders', 'xSpidersTOzSpiders', 'zSpiders'].edge_index = torch.tensor(coo_matrix.astype(np.int64))
    torch_data['xSpiders', 'xSpidersTOzSpiders', 'zSpiders'].edge_attr = torch.tensor(edge_features.astype(np.float32))
    

    if len(z_spider_indices) == 0:
        torch_data['zSpiders'].x = torch.tensor(0, dtype=torch.float32).reshape(-1,1)

    if len(x_spider_indices) == 0:
        torch_data['xSpiders'].x = torch.tensor(0, dtype=torch.float32).reshape(-1,1)

    #label_array = np.zeros((1,2), dtype=np.float32)
    #label_array[0, int(label)] = 1
    #torch_data.y = torch.tensor(label_array)

    return torch_data