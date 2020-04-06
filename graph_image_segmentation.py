import math
import networkx as nx
import numpy as np
from scipy import ndimage
from imageio import imread, imwrite
from sys import argv
import math
import random

class Component:
    def __init__(self, rep_id, size):
        self.rep_id = rep_id
        self.size = size
        self.int = math.inf
        self.color = self._rnd_color()

    def __eq__(self, other):
        if other is None:
            return False
        else: 
            return self.rep_id == other.rep_id

    def _rnd_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class DisjointSet:
    def __init__(self, id):
        self.id = id
        self.p = self
        self.rank = 0
        self._comp = None
        self.get_component()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.id == other.id

    def __ne__(self, other):
        if type(self) != type(other):
            return True
        return self.id != other.id

    def __hash__(self):
        return hash((self.id, self.rank))

    def find_set(self):
        if (self != self.p):
            self.p = self.p.find_set()
        return self.p

    def _update_parent(self, parent):
        old_comp_size = self.get_component().size
        self.p = parent
        self.get_component().size += old_comp_size

    def _link(self, other):
        if self.rank > other.rank:
            other._update_parent(self)
        elif other.rank > self.rank:
            self._update_parent(other)
        else:
            self.rank = self.rank + 1
            other._update_parent(self)

    def union(self, other):
        self.find_set()._link(other.find_set())

    def get_component(self):
        node_set = self.find_set()
        if self == node_set:
            if self._comp == None:
                self._comp = Component(self.id, 1)
            return self._comp
        else:
            self._comp = None
            return node_set.get_component()

def make_set(id):
    return DisjointSet(id)

def min_internal_diff(n_i, n_j, k):
    c_i, c_j = n_i.get_component(), n_j.get_component()
    comp_value = lambda c: c.int + k/c.size
    mint = min(comp_value(c_i), comp_value(c_j))
    return mint

def get_image_graph(filepath, sigma):
    im = ndimage.gaussian_filter(imread(filepath,  pilmode='L'), sigma, output='int')

    adj = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if not (i == j == 0)]
    # int_diff = lambda a, b: math.sqrt(
    #     math.pow(im[a[0], a[1], 0] - im[b[0], b[1], 0],2 ) +
    #     math.pow(im[a[0], a[1], 1] - im[b[0], b[1], 1],2 ) +
    #     math.pow(im[a[0], a[1], 2] - im[b[0], b[1], 2],2 ),  
    # )
    int_diff = lambda a, b: math.sqrt(
        abs(im[a[0], a[1]] - im[b[0], b[1]])
    )
    print(im.shape)
    rows, cols = im.shape
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            for dr, dc in adj:
                sr, sc = r + dr, c + dc
                if 0 <= sr < rows and 0 <= sc < cols:
                    _int = int_diff((r,c), (sr,sc))
                    G.add_edge( (r,c), (sr, sc), weight=_int)
    return G

def draw_segmentation(G,shape):
    if (len(shape) == 3):
        rows, cols, _ = shape
    else:
        rows, cols = shape 
    im = np.zeros((rows,cols, 3))
    for (r,c), rgb in nx.get_node_attributes(G, 'component').items():
        im[r,c, 0] = rgb[0]
        im[r,c, 1] = rgb[1]
        im[r,c, 2] = rgb[2]
    imwrite('z.jpg', im)


def segment(G, sigma=0.75, k=1): 
    sorted_edges = sorted(G.edges(data='weight'), key=lambda e: e[2])
    forest = {n: make_set(n) for n in G.nodes()}

    for id_i, id_j,w_q in sorted_edges:
        n_i, n_j = forest[id_i], forest[id_j]

        if (n_i.find_set() != n_j.find_set()) and w_q < min_internal_diff(n_i, n_j, k):
            n_i.union(n_j)
            n_i.get_component().int = w_q

    cs =  {k: v.get_component().color for k,v in forest.items()}
    print( set(cs.values()) )
    nx.set_node_attributes(G, cs, 'component')
    return G

def rnd_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# src: CLRS


def test_graph():
    G = nx.Graph()
    G.add_edge(0, 1, weight=4)
    G.add_edge(0, 7, weight=8)
    G.add_edge(1, 7, weight=11)
    G.add_edge(1, 2, weight=8)
    G.add_edge(7, 8, weight=7)
    G.add_edge(7, 6, weight=1)

    G.add_edge(2, 8, weight=2)
    G.add_edge(8, 6, weight=6)
    G.add_edge(2, 3, weight=7)
    G.add_edge(2, 5, weight=4)
    G.add_edge(6, 5, weight=2)
    G.add_edge(3, 5, weight=14)
    G.add_edge(3, 4, weight=9)
    G.add_edge(5, 4, weight=10)
    # G.add_edge(50, 51, weight=1000)
    # G.add_edge(50, 2, weight=1000)
    # G.add_edge(52, 51, weight=1003)

    return G

if __name__ == "__main__":

    G = test_graph()
    input_im = imread(argv[1])
    G = segment(get_image_graph(argv[1], 0))
    draw_segmentation(G, input_im.shape)
   # segment(G)
