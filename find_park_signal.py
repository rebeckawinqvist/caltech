import math

def dijkstras(ctrl, start):

    # Initializing set of visited nodes
    visited = set()
    states = set(ctrl.nodes())
    if start != 'Sinit':
        states.remove('Sinit')

    # Initializing distance from 'start' to any state in controller
    dist = dict.fromkeys(states, math.inf)
    # Previously visited states
    previous = dict.fromkeys(states, None)


    dist[start] = 0

    # While all states haven't been visited
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
    """Provided a controller and a desired transition for the system, return the
    optimal park signal sequence that triggers the transition

    des_transition - tuple/list with desired transition between two states in the system.
    des_transition[0] - start
    des_transition[1] - end

    """
    print("in here")
    to_visit = set()
    for edge in ctrl.edges_iter(data=True):
        next_ctrl_state = edge[1]  # end state of transition in controller
        neighbors = ctrl.neighbors(next_ctrl_state)
        neighbors_end_locs = []
        for neighbor in neighbors:
            neighbors_end_locs.append(ctrl.get_edge_data(next_ctrl_state, neighbor)[0]['loc'])
        sys_state = edge[2]['loc'] # controller steers into this system state

        if sys_state == des_transition[0] and des_transition[1] in neighbors_end_locs :  # if system state is the desired state
            to_visit.add(next_ctrl_state)


    dist, previous = dijkstras(ctrl, 'Sinit')
    shortest_path = []
    path_len = math.inf

    for state in to_visit:
        path = []
        while state != None:
            path.append(state)
            state = previous[state]

        if len(path) < path_len:
            shortest_path = path
            path_len = len(shortest_path)

    shortest_path.reverse()
    start = shortest_path[-1]
    neighbors = ctrl.neighbors(start)
    # Find neighbors that go to the correct state
    correct = []
    for neighbor in neighbors:
        if ctrl.get_edge_data(start, neighbor)[0]['loc'] == des_transition[1]:
            correct.append(neighbor)


    # Find neighbor that gives the shortest loop
    shortest_loop = []
    loop_len = math.inf

    for neighbor in correct:
        loop = []
        dist, previous = dijkstras(ctrl, neighbor)
        state = start
        while state != None:
            loop.append(state)
            state = previous[state]

        if len(loop) < loop_len:
            shortest_loop = loop
            loop_len = len(shortest_loop)

    shortest_loop.reverse()
    shortest_path = shortest_path + shortest_loop
    #repeat loop
    left = T-len(shortest_path)
    if left <= 0:
        seq = []
        for i in range(T-1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
            park_signal = ctrl.get_edge_data(u, v)[0]['park']
            seq.append(park_signal)
        return seq
    div = left//len(shortest_loop)
    rest = left%len(shortest_loop)
    for i in range(div):
        shortest_path += shortest_loop
    for i in range(rest+1):
        shortest_path.append(shortest_loop[i])

    seq = []
    for i in range(len(shortest_path)-1):
        u = shortest_path[i]
        v = shortest_path[i+1]
        park_signal = ctrl.get_edge_data(u,v)[0]['park']
        seq.append(park_signal)

    return seq








