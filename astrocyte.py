import networkx as nx



class Astrocyte:
    def __init__(self, image, graph, sigma_mask, params):
        self.graph = graph
        self.image = image
        self.sigma_mask = sigma_mask
        self.parameters = params

