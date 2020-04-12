import math
import random
from itertools import product, chain
from functools import partial, reduce
from sys import argv
from multiprocessing import Pool
import ntpath
import heapq 

# dependencies
import networkx as nx
import numpy as np
from scipy import ndimage
from imageio import imread, imwrite

class Component:
    def __init__(self, rep_id, size):
        self.rep_id = rep_id
        self.size = size
        self.int = math.inf
        self.color = rnd_color()

    def __eq__(self, other):
        if other is None:
            return False
        else: 
            return self.rep_id == other.rep_id

    def __hash__(self):
        return hash((self.rep_id, self.size, self.int))

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

    # def __ne__(self, other):
    #     if type(self) != type(other):
    #         return True
    #     return self.id != other.id

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
        else:
            self._update_parent(other)
            if self.rank == other.rank:
                self.rank += 1

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

# Tools
ADJ = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if not (i == j == 0)]
RGB = ['r', 'g' 'b']

def rnd_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def int_diff(pixel_a, pixel_b, im):
    return math.sqrt(abs(im[pixel_a] - im[pixel_b]))

def int_diff_color(pixel_a, pixel_b, im):
    return math.sqrt(sum( math.pow(im[pixel_a+(i,)] - im[pixel_b+(i,)], 2) for i in range(3)))

def get_edge(pos, im):
    shape = im.shape
    rows, cols = shape[0], shape[1]
    adj_edges = []

    for nbr in [(pos[0]+d[0], pos[1]+d[1]) for d in ADJ]:

        if 0 <= nbr[0] < rows and 0 <= nbr[1] < cols:
            if len(shape) == 2:
                weight = {"intensity_diff": int_diff(pos, nbr, im)}
            else:
                weight = {"intensity_diff": int_diff_color(pos, nbr, im)}
            adj_edges.append((pos, nbr, weight))

    return adj_edges

def get_image_graph(mode, sigma=0):
    im = ndimage.gaussian_filter(imread(filepath,  pilmode=mode), sigma)
    shape = im.shape
    print("image shape:", im.shape)

    G = nx.Graph()
    f = partial(get_edge, im=im)
    itr = product(range(shape[0]), range(shape[1]))
    l = pool.map(f, itr)
    G.add_edges_from(chain(*l))
    return G

def draw_segmentation(G, shape):
    if (len(shape) == 3):
        rows, cols, _ = shape
    else:
        rows, cols = shape 
    im = np.zeros((rows,cols, 3)).astype(np.uint8)
    for (r,c), rgb in nx.get_node_attributes(G, 'component').items():
        im[r,c, 0] = rgb[0]
        im[r,c, 1] = rgb[1]
        im[r,c, 2] = rgb[2]
    imwrite('z.jpg', im)


# Segmentation from NX graph
def make_set(id):
    return DisjointSet(id)

def min_internal_diff(n_i, n_j, k):
    c_i, c_j = n_i.get_component(), n_j.get_component()
    comp_value = lambda c: c.int + k/c.size
    mint = min(comp_value(c_i), comp_value(c_j))
    return mint

def segment(G, k, weight_key):
    sorted_edges = sorted(G.edges(data=weight_key), key=lambda e: e[2])
    forest = {n: make_set(n) for n in G.nodes()}

    for id_i, id_j, w_q in sorted_edges:
        n_i, n_j = forest[id_i], forest[id_j]

        if (n_i.find_set() != n_j.find_set()) and w_q <= min_internal_diff(n_i, n_j, k):
            n_i.union(n_j)
            n_i.get_component().int = w_q

    return {k: v.get_component().color for k,v in forest.items()}

def segment_and_update_graph(G, k=1):
    cs = segment(G, k, "intensity_diff")
    nx.set_node_attributes(G, cs, 'component')


# def distance(data, query):
#     return np.sqrt(np.square(data-query).sum(axis=1))

# def knn(data, k):
#     edges = []
#     print("finding knn")
#     for i in range(len(data)):
#         distances = distance(data, data[i])
        
#         heap = [(d, tuple(data[j])) for j, d in enumerate(distances) if i != j]
#         heapq.heapify(heap)     
#         nearest_nbrs = [heapq.heappop(heap)[1] for _ in range(k)]

#     print('done')

# def flatten_im_matrix(im):
#     row, cols, color = im.shape
#     coords = list(product(range(row), range(cols)))
#     m = np.zeros( (len(coords),5))
#     for i, (r,c) in enumerate(coords):
#         m[i] = [r,c, im[r,c,0], im[r,c,1], im[r,c,2]]
#     print(m)
#     knn(m, 10)
#     return m

if __name__ == "__main__":
    pool = Pool()

    filepath = argv[1]
    mode = "RGB" if argv[2] == "color" else "L"
    
    input_im = imread(filepath,  pilmode=mode)

    G = get_image_graph(mode, sigma=0.8)

    segment_and_update_graph(G, k=3000)

    draw_segmentation(G, input_im.shape)

    #print(flatten_im_matrix(input_im))