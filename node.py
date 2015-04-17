import itertools as it
import networkx as nx
import matplotlib.pylab as plt

class Node:
    def __init__(self, name=None, first=None, second=None, parent=None, value=None):
        self.name = name
        self.first = first
        if first:
            first.set_parent(self)
        self.second = second
        if second:
            second.set_parent(self)
        self.parent = parent
        self.value = value
        self.count = None

    def set_parent(self, parent):
        self.parent = parent

    def set_value(self, value):
        self.value = value

    def edge_list(self, counter=it.count().next):
        self.count = counter() if self.count is None else self.count
        for node in (self.first, self.second):
            if node:
                node.count = counter() if node.count is None else node.count
                yield (self.count, node.count)
        for node in (self.first, self.second):
            if node:
                for n in node.edge_list(counter):
                    yield n
    def label_list(self, counter=it.count().next):
        self.count = counter() if self.count is None else self.count
        yield (self.count, self.name)
        for node in (self.first, self.second):
            if node:
                for n in node.label_list(counter):
                    yield n

    def draw(self):
        edgelist = list(self.edge_list())
        G = nx.DiGraph(edgelist)
        pos = nx.graphviz_layout(G, prog='dot', root=0)
        nx.draw(G, pos)

        labellist = list(self.label_list())
        labels = {}
        for node,label in labellist:
            if label:
                labels[node] = label
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()
