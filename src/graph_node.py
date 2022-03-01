class GraphNode:
    def __init__(self, name):
        self.name = name
        self.input_layers = []
        self.output_layers = []

    def add_input_layer(self, layer):
        self.input_layers.append(layer)

    def add_output_layer(self, layer):
        self.output_layers.append(layer)

    def get_input_layers(self):
        return self.input_layers

    def get_output_layers(self):
        return self.output_layers
