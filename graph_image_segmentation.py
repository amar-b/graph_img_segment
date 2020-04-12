import math
import random
from itertools import product, chain
from functools import partial, reduce
from sys import argv
from multiprocessing import Pool
import warnings

# dependencies
import networkx as nx
import numpy as np
import cv2
from imageio import imread, imwrite
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# python-colormath
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

    # def __eq__(self, other):
    #     return self.id == other.id

    def __ne__(self, other):
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

def int_diff_color(pixel_a, pixel_b, lab):
  #  print(lab)
    return math.sqrt(sum( math.pow(c[pixel_a] - c[pixel_b], 2) for c in lab))

def get_edge(pos, shape, lab):
    rows, cols = shape[0], shape[1]
    adj_edges = []
    is_color = len(shape) == 3
    for nbr in [(pos[0]+d[0], pos[1]+d[1]) for d in ADJ]:

        if 0 <= nbr[0] < rows and 0 <= nbr[1] < cols:
            if is_color:
                weight = {"intensity_diff": int_diff_color(pos, nbr, lab)}
            else:
                weight = {"intensity_diff": int_diff(pos, nbr, lab)}
            adj_edges.append((pos, nbr, weight))

    return adj_edges

def get_image_graph(mode, sigma=0):
   # im = ndimage.gaussian_filter(imread(filepath,  pilmode=mode), sigma).astype(np.float64)
    im = np.float32(cv2.imread(filepath, -1))/255.
    im = cv2.GaussianBlur(im,(5,5),0)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2Lab)

    shape = im.shape
    print("image shape:", im.shape)

    G = nx.Graph()
    f = partial(get_edge, shape = shape, lab= cv2.split(im))
    itr = product(range(shape[0]), range(shape[1]))
   # print(len(list(itr)))
    l = pool.map(f, itr)
    G.add_edges_from(chain(*l))
    print(type(G))
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
    for id_i, id_j, w_q in sorted_edges:
        n_i, n_j = forest[id_i], forest[id_j]
        if n_i.find_set() != n_j.find_set() and min(n_i.get_component().size, n_j.get_component().size) <100:
            n_i.union(n_j)

    return {k: v.get_component().color for k,v in forest.items()}

def segment_and_update_graph(G, k=1):
    cs = segment(G, k, "intensity_diff")
    nx.set_node_attributes(G, cs, 'component')

if __name__ == "__main__":
    pool = Pool()

    filepath = argv[1]
    mode = "RGB" if argv[2] == "color" else "L"
    
    input_im = imread(filepath,  pilmode=mode)

    G = get_image_graph(mode, sigma=0.9)

    segment_and_update_graph(G, k=400)

    draw_segmentation(G, input_im.shape)

    # https://en.wikipedia.org/wiki/Color_difference
    #print(flatten_im_matrix(input_im))