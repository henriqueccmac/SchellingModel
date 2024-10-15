import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Schelling:
    def __init__(
        self,
        n_agent_classes,
        tolerance_threshold,
        lattice_m,
        lattice_n,
        empty_ratio,
        seed=0,
        z=5,
        delta_f=0.1,
        fmin=0,
        fmax=1,
    ):
        self.seed = seed
        self.empty_ratio = empty_ratio
        self.tolerance_threshold = tolerance_threshold
        self.n_agent_classes = n_agent_classes
        self.z = z  # Set z here
        self.delta_f = delta_f  # Change in tolerance
        self.fmin = fmin  # Minimum tolerance
        self.fmax = fmax  # Maximum tolerance

        self.lattice_m = lattice_m
        self.lattice_n = lattice_n
        self.graph = nx.grid_2d_graph(lattice_m, lattice_n)
        self.graph = self.add_diagonal_edges(self.graph)

        self.population_distribution = np.full(
            n_agent_classes + 1, (1 - empty_ratio) / n_agent_classes
        )
        self.population_distribution[0] = empty_ratio

        self.random_seeded = np.random.default_rng(seed)
        self.assign_agents()

    def add_diagonal_edges(self, graph):
        # Add diagonal edges if needed
        return graph

    def assign_agents(self):
        """Assign agents to the graph according to empty ratio and number of agent classes."""
        total_nodes = self.graph.number_of_nodes()

        agent_id_list = self.random_seeded.choice(
            range(self.n_agent_classes + 1),
            size=total_nodes,
            p=self.population_distribution,
        )

        positions = list(self.graph.nodes)
        positions = [tuple(pos) for pos in positions]  # Ensure tuples
        self.random_seeded.shuffle(positions)

        for pos, agent_id in zip(positions, agent_id_list):
            self.graph.nodes[pos]["class"] = None if agent_id == 0 else agent_id
            self.graph.nodes[pos]["threshold"] = self.tolerance_threshold

    def get_vacant_nodes(self):
        return [
            tuple(node)
            for node in self.graph.nodes
            if self.graph.nodes[node]["class"] is None
        ]

    def move_agent(self, node):
        """Implements the movement rule for an agent based on the pseudo-code."""
        vacant_nodes = self.get_vacant_nodes()
        # z = min(len(vacant_nodes), 5)  # Or any other logic to dynamically determine z
        # Randomly select a subset of vacant nodes (z candidate locations)
        candidate_locations = [
            tuple(loc)
            for loc in self.random_seeded.choice(
                vacant_nodes, size=min(len(vacant_nodes), self.z), replace=False
            )
        ]

        L_star = []  # List of locations that meet the tolerance condition

        for l in candidate_locations:
            # Neighborhood of the candidate location
            neighbors = list(self.graph.neighbors(l))

            # Count in-group and out-group neighbors
            g = len(
                [
                    n
                    for n in neighbors
                    if self.graph.nodes[n]["class"] == self.graph.nodes[node]["class"]
                ]
            )
            d = len(
                [
                    n
                    for n in neighbors
                    if self.graph.nodes[n]["class"] != self.graph.nodes[node]["class"]
                    and self.graph.nodes[n]["class"] is not None
                ]
            )

            if d > 0:  # Avoid division by zero
                s = g / (g + d)  # Ratio of in-group agents
            else:
                s = 1  # If no out-group agents, max satisfaction

            if s >= self.graph.nodes[node]["threshold"]:
                L_star.append(l)

        if L_star:
            # Randomly choose one location from L* and move the agent there
            new_location = tuple(self.random_seeded.choice(L_star))
            self.graph.nodes[new_location]["class"] = self.graph.nodes[node]["class"]
            self.graph.nodes[node]["class"] = None  # Vacate the old location
        else:
            # No suitable location found, agent stays in place
            return

    def decision_rule(self, node):
        """Implements the decision rule for agent ai."""
        agent_class = self.graph.nodes[node]["class"]
        tolerance = self.graph.nodes[node]["threshold"]

        neighbors = list(self.graph.neighbors(node))
        outgroup_count = sum(
            1
            for neighbor in neighbors
            if self.graph.nodes[neighbor]["class"] != agent_class
        )

        # Agent is exposed to outgroup agents
        if outgroup_count > 0:
            # If agent is dissatisfied (ui,t = 0)
            if self.is_dissatisfied(node, outgroup_count):
                # Move agent (Algorithm 1)
                self.move_agent(node)
                # Tolerance remains the same
                self.graph.nodes[node]["threshold"] = tolerance
            else:
                # Increase tolerance by Δf up to fmax
                self.graph.nodes[node]["threshold"] = min(
                    tolerance + self.delta_f, self.fmax
                )
                # With a small probability, move even when satisfied
                if np.random.random() <= 0.01:
                    vacant_nodes = self.get_vacant_nodes()
                    candidate_locations = self.random_seeded.choice(
                        vacant_nodes, size=min(len(vacant_nodes), self.z), replace=False
                    )
                    new_location = tuple(self.random_seeded.choice(candidate_locations))
                    self.graph.nodes[new_location]["class"] = agent_class
                    self.graph.nodes[node]["class"] = None
        else:
            # Decrease tolerance by Δf down to fmin
            self.graph.nodes[node]["threshold"] = max(
                tolerance - self.delta_f, self.fmin
            )

    def best_neighbour(self, location, migrants_set):
        """Finds the best neighboring site of location excluding locations in migrants_set."""
        neighbors = set(self.graph.neighbors(location)) - migrants_set
        vacant_neighbors = {
            n for n in neighbors if self.graph.nodes[n]["class"] is None
        }

        if vacant_neighbors:
            best_location = None
            best_density = float("-inf")

            for neighbor in vacant_neighbors:
                local_population = len(
                    [
                        n
                        for n in self.graph.neighbors(neighbor)
                        if self.graph.nodes[n]["class"] is not None
                    ]
                )
                local_density = local_population / len(
                    list(self.graph.neighbors(neighbor))
                )

                if local_density > best_density:
                    best_density = local_density
                    best_location = neighbor

            return (
                best_location
                if best_location
                else np.random.choice(list(vacant_neighbors))
            )

        # If no vacant neighbors found, pick a random neighbor
        return np.random.choice(list(neighbors))

    def place_migrants(self, focal_location, num_migrants):
        """Places num_migrants around a given focal location."""
        migrants_set = {focal_location}

        while len(migrants_set) < num_migrants:
            best_neighbor = self.best_neighbour(focal_location, migrants_set)
            migrants_set.add(best_neighbor)

        return migrants_set

    def is_dissatisfied(self, node, outgroup_count):
        """Determines if an agent is dissatisfied based on its tolerance."""
        neighbors = list(self.graph.neighbors(node))
        ingroup_count = len(
            [
                n
                for n in neighbors
                if self.graph.nodes[n]["class"] == self.graph.nodes[node]["class"]
            ]
        )

        if (ingroup_count + outgroup_count) == 0:
            satisfaction_ratio = 1  # If no neighbors, consider it fully satisfied
            print("No neighbors found")
        else:
            satisfaction_ratio = ingroup_count / (ingroup_count + outgroup_count)
            print(f"Satisfaction ratio: {satisfaction_ratio}")

        return satisfaction_ratio < self.graph.nodes[node]["threshold"]

    def simulate(self, max_steps=1000):
        """Runs the Schelling segregation model for a given number of steps."""
        for step in range(max_steps):
            nodes = list(self.graph.nodes())
            self.random_seeded.shuffle(nodes)  # Randomize the order of agent decisions

            # Apply decision rules to all agents
            for node in nodes:
                if (
                    self.graph.nodes[node]["class"] is not None
                ):  # Only for agents, not empty spaces
                    self.decision_rule(node)

            # Optional: You can visualize or log the state at every step
            if step % 100 == 0:
                print(f"Step {step} completed")
                self.plot_grid(step)

        print("Simulation completed")

    def plot_grid(self, step):
        """Visualizes the current state of the grid."""
        agent_grid = np.zeros((self.lattice_m, self.lattice_n))

        for (x, y), data in self.graph.nodes(data=True):
            if data["class"] is not None:
                agent_grid[x][y] = data["class"]

        plt.imshow(agent_grid, cmap="tab20b", interpolation="nearest")
        plt.title(f"Step {step}")
        plt.colorbar(label="Agent Class")
        plt.show()

    def migrate_agents(self, num_waves, tmig):
        """Handles the arrival of migrant waves at discrete intervals."""
        migration_interval = int(0.9 * tmig / num_waves)
        for wave in range(1, num_waves + 1):
            if wave * migration_interval <= tmig:
                focal_location = self.select_focal_location()
                num_migrants = round(self.calculate_migrant_count(wave, num_waves))
                self.place_migrants(focal_location, num_migrants)
                print(f"Migrant wave {wave} placed at {focal_location}")

    # def print_node_counts(self):
    #     """Print the total number of nodes for each class."""

    #     print("Total nodes per class:")
    #     for agent_class, count in self.node_count_per_class.items():
    #         if agent_class is None:
    #             print(f"Empty Nodes: {count}")
    #         else:
    #             print(f"Class {agent_class}: {count}")


n_agent_classes = 2
tolerance_threshold = 0.3
lattice_m = 20
lattice_n = 20
empty_ratio = 0.1
max_steps = 1000

model = Schelling(
    n_agent_classes, tolerance_threshold, lattice_m, lattice_n, empty_ratio
)

# fig, ax = plt.subplots(figsize=(8, 8))


# def update(frame):
#     """Update function for each animation frame. Simulation ends when all nodes are satisfied."""

#     if not model.simulate():  # Simulate until all satisfied
#         ani.event_source.stop()
#         print(f"Simulation ended at generation {frame + 1}.")
#     model.draw_graph(ax)
#     plt.title(f"Generation {frame + 1}")


# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=None,
#     interval=1,
#     repeat=False,
#     save_count=0,
#     cache_frame_data=False,
# )
model.simulate(max_steps)
plt.show()
