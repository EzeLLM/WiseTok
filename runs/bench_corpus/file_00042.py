from abc import ABCMeta
import json

class Graph(object):
    def __init__(self, root=None):
        self.root = root
        self._vertices = []
        self._edges = []
        self.adj_dict = {}                      # VERTEX_TAG : VERTEX ADJ'S {"1": [2, 3], "2": [1, 3, 4]}
        self._num_of_vertices = 0
        if root is not None:
            self.add_vertex(root)

    # should be a Vertex object
    def add_vertex(self, to_add):
        self._vertices.append(to_add)
        self.adj_dict.update({to_add.tag:[]})
        self._num_of_vertices += 1

    # should be an Edge object
    def add_edge(self, edge):
        self._edges.append(edge)
        self.adj_dict[edge.v1.tag].append(edge.v2.tag)  # add adjacency
        self.adj_dict[edge.v2.tag].append(edge.v1.tag)  # to both vertices of the edge

    # print the adjacency dict
    def print_adjacency(self):
        print (json.dumps(self.adj_dict))

    #  ------- Getters
    def get_num_vertices(self):
        return self._num_of_vertices

    def get_adjacency_dict(self):
        return self.adj_dict

    def get_vertex_for_tag(self,tag):
        return self._vertices[int(tag)-1]

    # return the edge of (v1, v2) if exists. else, return False
    def get_edge(self, v1, v2):
        for edge in self._edges:
            if edge.is_equal(v1, v2):
                if not edge.blocked:
                    return edge
                return False

    # returns all adjacent vertices that are NOT blocked
    def get_adjacent_to(self, vertex):
        if isinstance(vertex, Graph.Vertex):
            vertex = vertex.tag
        adjacent = self.adj_dict[vertex]
        adjacent = filter(lambda v: self.get_edge(v, vertex) is not False, adjacent)  # filter out all blocked
        adjacent.sort()
        return adjacent


    # describes an edge on the graph
    class Edge:
        def __init__(self, v1, v2, w):
            self.v1 = v1
            self.v2 = v2
            self.weight = w
            self.blocked = False

        # if edge v1-v2 equals (by tag) to this edge
        def is_equal(self, v1, v2):
            return (v1 == self.v1.tag and v2 == self.v2.tag) or (v1 == self.v2.tag and v2 == self.v1.tag)
        def block(self):
            self.blocked = True

    # describes a vertex on the graph
    class Vertex:
        __metaclass__ = ABCMeta
        def __init__(self, tag=0):
            self.tag = tag


