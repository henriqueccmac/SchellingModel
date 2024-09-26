import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class Schelling:
    def __init__(self, n_agent_classes, tolerance_treshold, lattice_m, lattice_n, empty_ratio, seed=0):
        self.seed = seed
        self.empty_ratio = empty_ratio
        self.tolerance_treshold = tolerance_treshold
        self.n_agent_classes = n_agent_classes
        
        self.lattice_m = lattice_m
        self.lattice_n = lattice_n
        self.graph = nx.grid_2d_graph(lattice_m,lattice_n)

        self.population_distribution = np.full(n_agent_classes+1, (1-empty_ratio)/n_agent_classes)
        self.population_distribution[0] = empty_ratio
        # i.e. [empty_ratio, 0.33, 0.33, 0.33] for 3 n_agent_classes

        self.random_seeded = np.random.default_rng(seed)
        self.assign_agents()

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
            self.graph.nodes[pos]['id'] = None if agent_id == 0 else agent_id
            self.graph.nodes[pos]['threshold'] = self.tolerance_treshold


    def get_neighbors(self, node):
        """Returns a list of all neighbors of node"""
        # TODO: maybe pass radius r and look into all neighbours up to level r? 
        return list(self.graph.neighbors(node))
    
    def is_unsatisfied(self, node):
        """Checks if neighbors are at least node.treshold similar to node"""
        agent_id = self.graph.nodes[node]['id']
        if agent_id is None:
            return False

        neighborhood = self.get_neighbors(node)
        similar_count = 0
        occupied_count = 0

        for neighbor in neighborhood:
            neighbor_agent_id = self.graph.nodes[neighbor]['id']

            if neighbor_agent_id is not None: 
                occupied_count += 1
                if neighbor_agent_id == agent_id:  # Similar neighbor
                    similar_count += 1

        if occupied_count == 0:  # No neighbors
            return False

        satisfaction_ratio = similar_count / occupied_count

        # The agent is unsatisfied if the similarity ratio is below its threshold
        return satisfaction_ratio < self.graph.nodes[node]['threshold']
    
    def move_agent(self, node):
        """Move agent to a free position/node in graph"""
        available_positions = [n for n in self.graph.nodes if self.graph.nodes[n]['id'] is None]

        if not available_positions:
            return

        new_position = tuple(self.random_seeded.choice(available_positions))

        self.graph.nodes[new_position]['id'] = self.graph.nodes[tuple(node)]['id']
        self.graph.nodes[tuple(node)]['id'] = None

    def is_balanced(self):
        """Simulation ends when all nodes are satisfied"""
        for node in list(self.graph.nodes):
            if self.graph.nodes[node]['id'] is not None:
                if self.is_unsatisfied(node):
                    return False
        return True

    def simulate(self):
        """Move agents according to their satisfaction"""
        for node in list(self.graph.nodes):
            if self.graph.nodes[node]['id'] is not None:
                if self.is_unsatisfied(node):
                    self.move_agent(node)
       
    def draw_graph(self, ax):
        """Assign a color to each node and draws the graph"""
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}  
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3'] 

        agent_color_map = {i + 1: colors[i % len(colors)] for i in range(self.n_agent_classes)}  # Mapping of agent ID to color
        agent_color_map[None] = '#FFFFFF'  # Empty nodes are white

        node_color = [agent_color_map.get(self.graph.nodes[node].get('id'), '#FFFFFF') for node in self.graph.nodes()]

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

schelling = Schelling(n_agent_classes=6, tolerance_treshold=0.5, lattice_m=20, lattice_n=20, empty_ratio=0.32)

fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    """Update function for each animation frame. Simulation ends when all nodes are satisfied"""
    if not schelling.is_balanced():
        schelling.simulate()
        schelling.draw_graph(ax)
        plt.title(f"Generation {frame + 1}")

ani = animation.FuncAnimation(fig, update, frames=100, interval=500, repeat=False)

plt.show()