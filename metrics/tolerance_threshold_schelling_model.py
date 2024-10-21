import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform

DEBUG = False 

class Schelling:
    def __init__(self, n_agent_classes, tolerance_threshold, lattice_m, lattice_n, empty_ratio, seed=0):
        self.seed = seed
        self.empty_ratio = empty_ratio
        self.tolerance_threshold = tolerance_threshold
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
        self.stuck_nodes = set()  # Nodes that are unable to move (no free positions)
	
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
            self.graph.nodes[pos]['threshold'] = self.tolerance_threshold
            self.graph.nodes[pos]['satisfaction_ratio'] = 0  

    def get_neighbors(self, node):
        """Returns a list of all neighbors of node, ie all nodes that have an edge connected to it"""
        return list(self.graph.neighbors(node))
    
    def calculate_satisfaction(self, node):
        """Checks if neighbors are at least node.threshold similar to node"""

        agent_id = self.graph.nodes[node]['class']
        if agent_id is None:
            return False

        neighborhood = self.get_neighbors(node)
        similar_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] == agent_id)
        occupied_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] is not None)

        if occupied_count == 0:  # No neighbors
            return False

        satisfaction_ratio = similar_count / occupied_count
        self.graph.nodes[node]['satisfaction_ratio'] = satisfaction_ratio

        if DEBUG:
            print(f"Node at {node} (Class: {agent_id}) - Satisfaction ratio: {satisfaction_ratio}, Threshold: {self.graph.nodes[node]['threshold']}")
            print(f"Node {node} is ////////////////////////////// Satisfied: {satisfaction_ratio > self.graph.nodes[node]['threshold']}")

        return satisfaction_ratio 
    
    def move_agent(self, node):
        """Attempt to move an unsatisfied agent to an available position. If no positions are available, mark as stuck."""
        neighbor_positions = self.get_neighbors(node)
        available_positions = [n for n in neighbor_positions if self.graph.nodes[n]['class'] is None]

        if not available_positions:
            self.stuck_nodes.add(node)  # Mark node as stuck
            return False

        # Move to new position
        new_position = tuple(self.random_seeded.choice(available_positions))
        self.graph.nodes[new_position]['class'] = self.graph.nodes[tuple(node)]['class']
        self.graph.nodes[tuple(node)]['class'] = None
        return True 

    def simulate(self):
        """Simulate a step in the model. If all unsatisfied nodes are stuck, the simulation ends."""
        all_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['class'] is not None]
        self.random_seeded.shuffle(all_nodes)

        unsatisfied_nodes = 0
        stuck_count = 0

        for node in all_nodes:
            self.calculate_satisfaction(node)

            if self.graph.nodes[node]['satisfaction_ratio'] < self.graph.nodes[node]['threshold']:
                unsatisfied_nodes += 1
                if not self.move_agent(node):
                    stuck_count += 1

        # End the simulation if all unsatisfied nodes are stuck
        return not (stuck_count == unsatisfied_nodes and unsatisfied_nodes > 0) 

    def calculate_metrics(self):
        """Calculate segregation metrics for the current state of the system"""
        homogeneity = []
        morans_i = []

        # Local homogeneity for each agent
        for node in self.graph.nodes:
            if self.graph.nodes[node]['class'] is not None:
                neighborhood = self.get_neighbors(node)
                similar_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] == self.graph.nodes[node]['class'])
                occupied_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] is not None)
                if occupied_count > 0:
                    homogeneity.append(similar_count / occupied_count)

        homogeneity_score = np.mean(homogeneity)

        # Moran's I calculation for spatial autocorrelation
        agent_classes = np.array([self.graph.nodes[node]['class'] if self.graph.nodes[node]['class'] is not None else -1 for node in self.graph.nodes])
        dist_matrix = squareform(pdist(np.array(list(self.graph.nodes)), 'euclidean'))
        morans_i_score = self.morans_i(agent_classes, dist_matrix)

        return homogeneity_score, morans_i_score

    def morans_i(self, classes, dist_matrix):
        """Calculate Moran's I index to measure spatial autocorrelation."""
        weights = 1 / (dist_matrix + 1e-10)  # Add small value to avoid division by zero
        np.fill_diagonal(weights, 0)  # No self influence
        W = weights.sum()

        mean_classes = np.mean(classes)
        deviation = classes - mean_classes
        num = np.sum(np.outer(deviation, deviation) * weights)
        denom = np.sum(deviation ** 2)
        return (len(classes) / W) * (num / denom)
    
    def draw_graph(self, ax):
        """Assign a color to each node and draw the graph"""

        pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}  
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3'] 

        agent_color_map = {i + 1: colors[i % len(colors)] for i in range(self.n_agent_classes)}  # Mapping of agent ID to color
        agent_color_map[None] = '#FFFFFF'  # Empty nodes are white

        node_color = []
        for node in self.graph.nodes():
            if node in self.stuck_nodes:
                node_color.append('#000000')  # Stuck nodes are black 
            else:
                node_color.append(agent_color_map.get(self.graph.nodes[node].get('class'), '#FFFFFF'))

        ax.clear()

        no_edges = nx.subgraph_view(self.graph, filter_edge=nx.classes.filters.hide_edges(self.graph.edges))

        total_nodes = self.graph.number_of_nodes()
        base_node_size = 100  
        scaling_factor = 1000 / total_nodes  # Scale down the node size based on total nodes
        adjusted_node_size = base_node_size * scaling_factor

        nx.draw(
            no_edges,
            pos,
            node_color=node_color,
            with_labels=False,
            node_size=adjusted_node_size,
            edge_color=None,
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

threshold_values = np.linspace(0.3, 0.95, 50)  
final_homogeneity_values = []
final_morans_i_values = []

def simulate_for_threshold(threshold):
    """Run the simulation for a specific tolerance threshold without animation."""
    schelling = Schelling(n_agent_classes=2, tolerance_threshold=threshold, lattice_m=50, lattice_n=50, empty_ratio=0.50, seed=0)

    max_frames = 1000
    frames = 0
    while frames < max_frames and schelling.simulate():
        frames += 1  

    return schelling.calculate_metrics()

for threshold in threshold_values:
    print(f"Simulating for threshold: {threshold:.2f}")
    
    final_homogeneity, final_morans_i = simulate_for_threshold(threshold)
    
    final_homogeneity_values.append(final_homogeneity)
    final_morans_i_values.append(final_morans_i)

# Plotting after all simulations
plt.figure()
plt.plot(threshold_values, final_homogeneity_values, marker='o')
plt.title("Homogeneity vs. Threshold")
plt.xlabel("Tolerance Threshold")
plt.ylabel("Homogeneity")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(threshold_values, final_morans_i_values, marker='x')
plt.title("Moran's I vs. Threshold")
plt.xlabel("Tolerance Threshold")
plt.ylabel("Moran's I")
plt.grid(True)
plt.show()
