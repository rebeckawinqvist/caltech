import collections
import math


class Graph:
    ''' graph class inspired by https://gist.github.com/econchick/4666413
    '''

    def __init__(self):
        self.vertices = set()

        # makes the default value for all vertices an empty list
        self.edges = collections.defaultdict(list)
        self.weights = {}

    def add_vertex(self, value):
        self.vertices.add(value)

    def add_edge(self, from_vertex, to_vertex, distance):
        if from_vertex == to_vertex: pass  # no cycles allowed
        self.edges[from_vertex].append(to_vertex)
        self.weights[(from_vertex, to_vertex)] = distance

    def __str__(self):
        string = "Vertices: " + str(self.vertices) + "\n"
        string += "Edges: " + str(self.edges) + "\n"
        string += "Weights: " + str(self.weights)
        return string


def dijkstra(graph, start):
    # initializations
    S = set()

    # delta represents the length shortest distance paths from start -> v, for v in delta.
    # We initialize it so that every vertex has a path of infinity (this line will break if you run python 2)
    delta = dict.fromkeys(list(graph.vertices), math.inf)
    print("delta: ", delta)
    previous = dict.fromkeys(list(graph.vertices), None)

    # then we set the path length of the start vertex to 0
    delta[start] = 0

    # while there exists a vertex v not in S
    while S != graph.vertices:
        # let v be the closest vertex that has not been visited...it will begin at 'start'
        print("keys: ", set(delta.keys())-S)
        v = min((set(delta.keys()) - S), key=delta.get)

        # for each neighbor of v not in S
        for neighbor in set(graph.edges[v]) - S:
            new_path = delta[v] + graph.weights[v, neighbor]

            # is the new path from neighbor through
            if new_path < delta[neighbor]:
                # since it's optimal, update the shortest path for neighbor
                delta[neighbor] = new_path

                # set the previous vertex of neighbor to v
                previous[neighbor] = v
        S.add(v)

    return (delta, previous)


def shortest_path(graph, start, end):
    '''Uses dijkstra function in order to output the shortest path from start to end
    '''


    delta, previous = dijkstra(graph, start)

    path = []
    vertex = end

    while vertex is not None:
        path.append(vertex)
        vertex = previous[vertex]

    path.reverse()
    return path


G = Graph()
G.add_vertex('a')
G.add_vertex('b')
G.add_vertex('c')
G.add_vertex('d')
G.add_vertex('e')

G.add_edge('a', 'b', 2)
G.add_edge('a', 'c', 8)
G.add_edge('a', 'd', 5)
G.add_edge('b', 'c', 1)
G.add_edge('c', 'e', 3)
G.add_edge('d', 'e', 4)

print(G)

print(dijkstra(G, 'a'))

#########################################
import math

def dijkstras(ctrl, start):

    #initializing set of visited nodes
    visited = set()
    states = set(ctrl.nodes())
    if start != 'Sinit':
        states.remove('Sinit')

    #distance from 'Sinit' to any state in controller
    dist = dict.fromkeys(states, math.inf)
    #previously visited state
    previous = dict.fromkeys(states, None)

    #'Sinit' is init state
    #start = 'Sinit'

    dist[start] = 0

    #while all states haven't been visited
    while visited != states:
        state = min((set(dist.keys())-visited), key=dist.get)
        for neighbor in set(ctrl.neighbors(state)):
            path = dist[state] + 1

            if path < dist[neighbor]:
                dist[neighbor] = path
                previous[neighbor] = state

        visited.add(state)

    return(dist, previous)



def find_sequence(ctrl, des_transition, T):
    """Provided a controller and a desired state for the system, return the
    optimal park signal sequence that forces the system into the desired state

    des_transition - tuple/list with desired transition between two states in the system.
    des_transition[0] - start
    des_transition[1] - end

    """
    to_visit_first = set()
    to_visit_second = set()
    for edge in ctrl.edges_iter(data=True):
        next_ctrl_state = edge[1]  # end state of transition in controller
        sys_state = edge[2]['loc'] # controller steers into this system state
        #print("sys_state: ", sys_state)
        if sys_state == des_transition[0]:  # if system state is the desired state
            to_visit_first.add(next_ctrl_state)  # want to visit this controller state
        elif sys_state == des_transition[1]:
            to_visit_second.add(next_ctrl_state)


    print("first: ", to_visit_first)
    print("second: ", to_visit_second)

    dist, previous = dijkstras(ctrl, 'Sinit')

    shortest_path = []
    path_len = math.inf

    print("to visit in ctrl: ", to_visit_first)
    for first_state in to_visit_first:
        first = first_state

        first_path = []
        while first_state != None:
            first_path.append(first_state)
            first_state = previous[first_state]

        first_path.reverse()
        print("first path: ", first_path)
        dist2, previous2 = dijkstras(ctrl, first)

        for second_state in to_visit_second:
            second_path = []
            while second_state != None:
                second_path.append(second_state)
                second_state = previous2[second_state]
            second_path.reverse()
            second_path = second_path[1:]
            print("second path: ", second_path)
            path = first_path + second_path

            if len(path) < path_len:
                shortest_path = path
                path_len = len(shortest_path)


    #reverse path
    print("shortest: ", shortest_path)

    #shortest_path.append()
    seq = []
    for i in range(len(shortest_path)-1):
        u = shortest_path[i]
        v = shortest_path[i+1]
        park_signal = ctrl.get_edge_data(u,v)[0]['park']
        seq.append(park_signal)

    return(seq)








