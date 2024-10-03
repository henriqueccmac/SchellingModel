import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ideas:
# change the probability distribution of the classes
# increase neighborhood of nodes
# if at the end of simulation there unsatisfied nodes because they cant move, change their color

class Schelling:
    def __init__(self, n_agent_classes, tolerance_treshold, lattice_m, lattice_n, empty_ratio, seed=0):
        self.seed = seed
        self.empty_ratio = empty_ratio
        self.tolerance_treshold = tolerance_treshold
        self.n_agent_classes = n_agent_classes
        
        self.lattice_m = lattice_m
        self.lattice_n = lattice_n
        self.graph = nx.grid_2d_graph(lattice_m,lattice_n)
        self.graph = self.add_diagonal_edges(self.graph)

        self.population_distribution = np.full(n_agent_classes+1, (1-empty_ratio)/n_agent_classes)
        self.population_distribution[0] = empty_ratio
        # i.e. [empty_ratio, 0.33, 0.33, 0.33] for 3 n_agent_classes

        self.random_seeded = np.random.default_rng(seed)
        self.assign_agents()
        self.node_count_per_class = self.count_nodes_per_class()
	
    def add_diagonal_edges(self, graph):
        """Add diagonal edges to existing grid graph"""
        for x in range(self.lattice_m):
            for y in range(self.lattice_n):
                node = (x,y)
                diagonal_neighbors = { (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1) }
                for n in diagonal_neighbors:
                    if n in graph.nodes:
                        graph.add_edge(node, n)
        return graph
    
    def assign_agents(self):
        """Assigns agents to the graph according to empty ratio and number of agent classes"""
        total_nodes = self.graph.number_of_nodes()
        
        # Possible agents: [0 (empty), 1, 2, ..., n_agents]
        agent_id_list = self.random_seeded.choice(
            range(self.n_agent_classes + 1),
            size=total_nodes,
            p=self.population_distribution
        )
        
        # Ensure agents are assigned randomly
        positions = list(self.graph.nodes)
        self.random_seeded.shuffle(positions)

        for pos, agent_id in zip(positions, agent_id_list):
            self.graph.nodes[pos]['class'] = None if agent_id == 0 else agent_id
            self.graph.nodes[pos]['threshold'] = self.tolerance_treshold
            self.graph.nodes[pos]['movable'] = True

    def get_neighbors(self, node):
        """Returns a list of all neighbors of node, ie all nodes that have an edge connected to it"""
        return list(self.graph.neighbors(node))
    
    def is_unsatisfied(self, node):
        """Checks if neighbors are at least node.treshold similar to node"""
        if self.graph.nodes[node]['movable'] == False:
            # Cant move, no point in trying to calculate satisfaction to see if will move
            return False       

        agent_id = self.graph.nodes[node]['class']
        if agent_id is None:
            return False

        neighborhood = self.get_neighbors(node)
        similar_count = 0
        occupied_count = 0

        for neighbor in neighborhood:
            neighbor_agent_id = self.graph.nodes[neighbor]['class']

            if neighbor_agent_id is not None: 
                occupied_count += 1
                if neighbor_agent_id == agent_id:  # Similar neighbor
                    similar_count += 1

        if occupied_count == 0:  # No neighbors
            return False

        satisfaction_ratio = similar_count / occupied_count

        # Debug log
        print(f"Node at {node} (Class: {agent_id}) - Satisfaction ratio: {satisfaction_ratio}, Threshold: {self.graph.nodes[node]['threshold']}")

        # The agent is unsatisfied if the similarity ratio is below its threshold
        return satisfaction_ratio < self.graph.nodes[node]['threshold']
    
    def move_agent(self, node):
        """Move agent to any free position/node in neighborhood"""
        if self.graph.nodes[node]['movable'] == False:
            return
        
        neighbor_positions = self.get_neighbors(node)

        available_positions = [n for n in neighbor_positions if self.graph.nodes[n]['class'] is None]

        if not available_positions:
            self.graph.nodes[node]['movable'] = False # surrounded by non similar class, cant move
            return

        new_position = tuple(self.random_seeded.choice(available_positions))

        self.graph.nodes[new_position]['class'] = self.graph.nodes[tuple(node)]['class']
        self.graph.nodes[tuple(node)]['class'] = None

    def is_balanced(self):
        """Simulation ends when all nodes are satisfied"""
        for node in list(self.graph.nodes):
            if self.is_unsatisfied(node):
                return False
        return True

    def simulate(self):
        """Move agents according to their satisfaction needs"""
        for node in list(self.graph.nodes):
            if self.graph.nodes[node]['class'] is not None:
                if self.is_unsatisfied(node):
                    self.move_agent(node)
       
    def draw_graph(self, ax):
        """Assign a color to each node and draw the graph"""
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}  
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3'] 

        agent_color_map = {i + 1: colors[i % len(colors)] for i in range(self.n_agent_classes)}  # Mapping of agent ID to color
        agent_color_map[None] = '#FFFFFF'  # Empty nodes are white

        node_color = [agent_color_map.get(self.graph.nodes[node].get('class'), '#FFFFFF') for node in self.graph.nodes()]

        ax.clear()

        nx.draw(
            self.graph,
            pos,
            node_color=node_color,
            with_labels=False,
            node_size=300,
            edge_color='gray',
            ax=ax
        )
    
    def print_node_counts(self):
        """Print the total number of nodes for each class."""
        print("Total nodes per class:")
        for agent_class, count in self.node_count_per_class.items():
            if agent_class is None:
                print(f"Empty Nodes: {count}")
            else:
                print(f"Class {agent_class}: {count}")

    def count_nodes_per_class(self):
        """Count total number of nodes for each class."""
        count = {i: 0 for i in range(1, self.n_agent_classes + 1)}
        count[None] = 0  # Count for empty spaces

        for node in self.graph.nodes:
            agent_id = self.graph.nodes[node]['class']
            if agent_id is not None:
                count[agent_id] += 1
            else:
                count[None] += 1
        return count

schelling = Schelling(n_agent_classes=2, tolerance_treshold=0.5, lattice_m=20, lattice_n=30, empty_ratio=0.3, seed=0)

fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    """Update function for each animation frame. Simulation ends when all nodes are satisfied"""
    if not schelling.is_balanced():
        schelling.simulate()
        schelling.draw_graph(ax)
        plt.title(f"Generation {frame + 1}")

schelling.print_node_counts()
ani = animation.FuncAnimation(fig, update, frames=500, interval=100, repeat=False)

plt.show()
