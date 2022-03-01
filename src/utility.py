from tensorflow.keras import backend as K

from src.graph_node import GraphNode


def compute_layer_output(layer_dependency_graph, model_layers, original_input, layer_output_dict, layer):
    """
    This function computes the layer output. It searches for the longest consecutive model portion
    to get prediction. It recursively computes input if any layer has multiple inputs. It then stores
    the layer output in the output dictionary.

    :param layer_dependency_graph: layer dependency graph of model
    :param model_layers: list of layers of the model
    :param original_input: original input tensor used for prediction
    :param layer_output_dict: a dictionary which stores outputs of different layer for memorization
    :param layer: the layer whose output needs to be calculated
    :return:
    """
    current_layer = layer
    input_tensor_list = []
    while True:
        inputs_of_current_layer = layer_dependency_graph[current_layer].get_input_layers()
        if len(inputs_of_current_layer) == 1:
            if inputs_of_current_layer[0] in layer_output_dict:
                input_tensor_list.append(layer_output_dict[inputs_of_current_layer[0]])
                start_layer = current_layer
                break
            else:
                current_layer = inputs_of_current_layer[0]
        else:
            for node in inputs_of_current_layer:
                if node not in layer_output_dict:
                    compute_layer_output(layer_dependency_graph, model_layers, original_input, layer_output_dict, node)
                input_tensor_list.append(layer_output_dict[node])
            start_layer = current_layer
            break
    get_pred = K.function([model_layers[start_layer].input], [model_layers[layer].output])
    layer_output_dict[layer] = get_pred(input_tensor_list)


def compute_fault_injected_prediction(layer_dependency_graph, super_nodes, model_layers,
                                      injection_layer_index, injected_output, original_input):
    """
    Calculate the model final predicted output using the fault injected output and original input.
    It stores the layer output results in a dictionary to get rid of recomputation. It initially
    computes prediction of immediate previous super layer and store in dict because all the branching
    starts from the previous super layer. It then calculates the inputs of next super layer recursively
    with memorization. Using those inputs, it finally calculates final prediction.

    :param layer_dependency_graph: layer dependency graph of model
    :param super_nodes: list of all super nodes
    :param model_layers: list of layers of the model
    :param injection_layer_index: the index where fault is injected
    :param injected_output: fault injected output
    :param original_input: original input tensor used for prediction
    :return: faulty model predicted output
    """
    layer_output_dict = {injection_layer_index: injected_output}
    if injection_layer_index not in super_nodes:
        previous_super_layer = get_previous_super_layer(layer_dependency_graph, super_nodes, injection_layer_index)
        pred_function = K.function([model_layers[0].input], [model_layers[previous_super_layer].output])
        layer_output_dict[previous_super_layer] = pred_function(original_input)

    next_super_layer = get_next_super_layer(layer_dependency_graph, super_nodes, injection_layer_index)
    next_super_layer_inputs = layer_dependency_graph[next_super_layer].get_input_layers()
    input_tensor_list = []
    for layer in next_super_layer_inputs:
        if layer not in layer_output_dict:
            compute_layer_output(layer_dependency_graph, model_layers, original_input, layer_output_dict, layer)
        input_tensor_list.append(layer_output_dict[layer])
    get_pred = K.function([model_layers[next_super_layer].input], [model_layers[-1].output])
    return get_pred(input_tensor_list)


def build_dependency_graph(model_layers, layer_name_to_index):
    """
    This function builds a dependency graph using the model layers. For each layer
    it stores the unique name of the output of that layer, the layer indices on which
    it depends and the layer indices which depend on this layer.

    :param model_layers: List of layers of the model
    :param layer_name_to_index: a dictionary that contains mapping of all layer output names to their index
    :return: a dependency graph
    """
    dependency_graph = []
    index = 0
    for layer in model_layers:
        graph_element = GraphNode(layer.output.name)
        if index != 0:
            if type(layer.input) is list:
                for layer_name in layer.input:
                    layer_index = layer_name_to_index[layer_name.name]
                    graph_element.add_input_layer(layer_index)
                    dependency_graph[layer_index].add_output_layer(index)
            else:
                layer_index = layer_name_to_index[layer.input.name]
                graph_element.add_input_layer(layer_index)
                dependency_graph[layer_index].add_output_layer(index)
        dependency_graph.append(graph_element)
        index += 1
    return dependency_graph


def map_layer_output_name_with_index(model_layers):
    """
    Each layer has a unique output name. This function maps this unique name with
    the layer index so that we can easily access each layer with index instead of name.

    :param model_layers: List of layers of the model
    :return: a dictionary that contains mapping of all layer names to their index
    """
    output_name_to_index = {}
    total_layer_count = len(model_layers)
    for i in range(total_layer_count):
        output_name_to_index[model_layers[i].output.name] = i
    return output_name_to_index


def get_super_nodes(layer_dependency_graph, start_node, end_node, super_node_dict):
    """
    Returns super nodes list from start_node position to end_node position of layer_dependency_graph

    :param layer_dependency_graph: dependency graph of the model
    :param start_node: starting layer index from where super node searching starts
    :param end_node: ending layer index of super node searching
    :param super_node_dict: a temporary dictionary which keeps track of already searched portion for super nodes
    :return:
    """
    super_nodes = [start_node]
    while start_node < end_node:
        graph_node = layer_dependency_graph[start_node]
        if len(graph_node.get_output_layers()) == 1:
            start_node = graph_node.get_output_layers()[0]
            super_nodes.append(start_node)
        else:
            # Recursively collect super nodes of each branch and combine them to get the overall supernode list
            partial_super_nodes_list = []
            for node in graph_node.get_output_layers():
                if node not in super_node_dict:
                    partial_super_nodes = get_super_nodes(layer_dependency_graph, node, end_node, super_node_dict)
                    super_node_dict[node] = partial_super_nodes
                partial_super_nodes_list.append(super_node_dict[node])
            super_nodes.extend(get_common_elements(partial_super_nodes_list))
            break
    return super_nodes


def get_common_elements(element_list):
    """
    :param element_list: list of list where each internal list contains values
    :return: a sorted list of elements which are common in all the internal lists
    """
    common_element_list = set(element_list[0])
    index = 1
    while index < len(element_list):
        common_element_list = common_element_list.intersection(element_list[index])
        index += 1
    return sorted(list(common_element_list))


def get_next_super_layer(layer_dependency_graph, super_nodes, current_layer):
    """
    Return the immediate next super layer of current layer

    :param layer_dependency_graph: dependency graph of the model
    :param super_nodes: list of all super nodes
    :param current_layer: the layer whose next super layer need to compute
    :return: immediate next super layer
    """
    current_layer = layer_dependency_graph[current_layer].get_output_layers()[0]
    while True:
        if current_layer in super_nodes:
            return current_layer
        current_layer = layer_dependency_graph[current_layer].get_output_layers()[0]


def get_previous_super_layer(layer_dependency_graph, super_nodes, current_layer):
    """
    Return the immediate previous super layer of current layer

    :param layer_dependency_graph: dependency graph of the model
    :param super_nodes: list of all super nodes
    :param current_layer: the layer whose previous super layer need to compute
    :return: immediate previous super layer
    """
    current_layer = layer_dependency_graph[current_layer].get_input_layers()[0]
    while True:
        if current_layer in super_nodes:
            return current_layer
        current_layer = layer_dependency_graph[current_layer].get_input_layers()[0]


def draw_graph(layer_dependency_graph, super_nodes, name):
    """
    This method is used to draw the directed dependency graph. In the graph
    super nodes are filled with red color. It is helpful to check whether the
    drawn dependency graph and super nodes are correct or not. It also eases
    the debugging process.

    :param layer_dependency_graph: graph generated from model indicating input and output layers.
    :param super_nodes: The nodes which are central to the model
    :param name: graph is being saved by this name
    :return: saves a pdf of the dependency graph
    """
    import pygraphviz as pgv

    graph = pgv.AGraph(directed=True)
    graph.node_attr['style'] = 'filled'
    graph.node_attr['shape'] = 'circle'
    graph.node_attr['fixedsize'] = 'true'
    graph.node_attr['fontcolor'] = '#000000'
    layer_len = len(layer_dependency_graph)

    for i in range(layer_len - 1):
        model_elem = layer_dependency_graph[i]
        for node in model_elem.get_output_layers():
            graph.add_edge(i, node)
    for node in super_nodes:
        n = graph.get_node(node)
        n.attr['fillcolor'] = "#FF0000"
    graph.draw(name + '.pdf', prog="circo")


def get_fault_injection_configs(model, graph_drawing=False):
    """
    For each model we need to develop a dependency graph of inputs and outputs of all the
    layers. We also compute the super nodes list. Dependency graph and super nodes list are
    model specific, so we can compute them before starting fault injection.

    :param model: keras model on which faults will be injected
    :param graph_drawing: determines whether a pdf version of dependency graph will be saved or not. Default=False
    :return: dependency graph and super node list
    """
    layer_name_to_index = map_layer_output_name_with_index(model.layers)
    dependency_graph = build_dependency_graph(model.layers, layer_name_to_index)
    super_nodes = get_super_nodes(dependency_graph, 0, len(model.layers) - 1, {})

    if graph_drawing:
        draw_graph(dependency_graph, super_nodes, model.name)
    return dependency_graph, super_nodes
