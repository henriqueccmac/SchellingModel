import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

DEBUG = True

class Schelling:
    def __init__(self, n_agent_classes, tolerance_threshold, lattice_m, lattice_n, empty_ratio, seed=0, tolerance_step=0.05):
        self.seed = seed
        self.empty_ratio = empty_ratio
        self.tolerance_threshold = tolerance_threshold
        self.tolerance_step = tolerance_step
        self.n_agent_classes = n_agent_classes
        
        self.lattice_m = lattice_m
        self.lattice_n = lattice_n
        self.graph = nx.grid_2d_graph(lattice_m, lattice_n)
        self.graph = self.add_diagonal_edges(self.graph)

        self.population_distribution = np.full(n_agent_classes + 1, (1 - empty_ratio) / n_agent_classes)
        self.population_distribution[0] = empty_ratio

        self.random_seeded = np.random.default_rng(seed)
        self.assign_agents()
        self.node_count_per_class = self.count_nodes_per_class()
        self.generation_count = 0  # Contador de gerações

    def add_diagonal_edges(self, graph):
        for x in range(self.lattice_m):
            for y in range(self.lattice_n):
                node = (x, y)
                diagonal_neighbors = {(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)}
                for n in diagonal_neighbors:
                    if n in graph.nodes:
                        graph.add_edge(node, n)
        return graph
    
    def assign_agents(self):
        total_nodes = self.graph.number_of_nodes()
        
        agent_id_list = self.random_seeded.choice(
            range(self.n_agent_classes + 1),
            size=total_nodes,
            p=self.population_distribution
        )
        
        positions = list(self.graph.nodes)
        self.random_seeded.shuffle(positions)

        for pos, agent_id in zip(positions, agent_id_list):
            self.graph.nodes[pos]['class'] = None if agent_id == 0 else agent_id
            self.graph.nodes[pos]['threshold'] = self.tolerance_threshold
            self.graph.nodes[pos]['satisfaction_ratio'] = 0 

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def calculate_satisfaction(self, node):
        agent_id = self.graph.nodes[node]['class']
        if agent_id is None:
            return False

        neighborhood = self.get_neighbors(node)
        similar_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] == agent_id)
        occupied_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] is not None)

        if occupied_count == 0:
            return False

        satisfaction_ratio = similar_count / occupied_count
        self.graph.nodes[node]['satisfaction_ratio'] = satisfaction_ratio

        return satisfaction_ratio

    def move_agent(self, node):
        neighbor_positions = self.get_neighbors(node)
        available_positions = [n for n in neighbor_positions if self.graph.nodes[n]['class'] is None]

        if not available_positions:
            return

        new_position = tuple(self.random_seeded.choice(available_positions))

        self.graph.nodes[new_position]['class'] = self.graph.nodes[tuple(node)]['class']
        self.graph.nodes[tuple(node)]['class'] = None

    def simulate(self):
        all_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['class'] is not None]
        self.random_seeded.shuffle(all_nodes)

        moved = False

        for node in all_nodes:
            self.calculate_satisfaction(node)

            if self.graph.nodes[node]['satisfaction_ratio'] < self.graph.nodes[node]['threshold']:
                self.adapt_tolerance(node, increase=False)
                self.move_agent(node)
                moved = True
            else:
                self.adapt_tolerance(node, increase=True)

        if moved:  # Incrementa o contador de gerações se houver movimento
            self.generation_count += 1

        return moved

    def adapt_tolerance(self, node, increase=True):
        satisfaction_ratio = self.graph.nodes[node]['satisfaction_ratio']
        neighborhood = self.get_neighbors(node)
        total_occupied_neighbors = sum(1 for n in neighborhood if self.graph.nodes[n]['class'] is not None)

        if total_occupied_neighbors == 0:
            return

        dissimilar_neighbors = total_occupied_neighbors * (1 - satisfaction_ratio)

        if increase:
            self.graph.nodes[node]['threshold'] += self.tolerance_step * dissimilar_neighbors
        else:
            self.graph.nodes[node]['threshold'] -= self.tolerance_step * dissimilar_neighbors

        self.graph.nodes[node]['threshold'] = max(0.2, min(0.98, self.graph.nodes[node]['threshold']))
    
    def draw_graph(self, ax):
        grid = np.zeros((self.lattice_m, self.lattice_n))

        for (x, y), data in self.graph.nodes(data=True):
            agent_class = data.get('class')
            grid[x, y] = agent_class if agent_class is not None else 0

        colors = ['#FFFFFF', '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3']
        cmap = plt.matplotlib.colors.ListedColormap(colors[:self.n_agent_classes + 1])

        ax.clear()
        ax.imshow(grid, cmap=cmap, origin='upper', extent=[0, self.lattice_n, 0, self.lattice_m])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    def print_node_counts(self):
        print("Total nodes per class:")
        for agent_class, count in self.node_count_per_class.items():
            if agent_class is None:
                print(f"Empty Nodes: {count}")
            else:
                print(f"Class {agent_class}: {count}")

    def count_nodes_per_class(self):
        count = {i: 0 for i in range(1, self.n_agent_classes + 1)}
        count[None] = 0

        for node in self.graph.nodes:
            agent_id = self.graph.nodes[node]['class']
            if agent_id is not None:
                count[agent_id] += 1
            else:
                count[None] += 1
        return count

    def average_satisfaction(self):
        satisfaction_ratios = [self.graph.nodes[node]['satisfaction_ratio'] for node in self.graph.nodes if self.graph.nodes[node]['class'] is not None]
        if not satisfaction_ratios:
            return 0
        return sum(satisfaction_ratios) / len(satisfaction_ratios)


class SchellingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Schelling Segregation Model")
        self.auto_simulating = False
        self.simulation_speed = 1

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.grid(row=0, column=0, columnspan=4, sticky='nsew')

        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=4, sticky='ew')

        self.root.grid_rowconfigure(0, weight=4)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self.close_program)

        self.reset_simulation()

    def setup_widgets(self):
        self.lattice_rows_label = tk.Label(self.control_frame, text="Lattice Rows:")
        self.lattice_rows_entry = tk.Entry(self.control_frame)
        self.lattice_rows_entry.insert(0, '20')

        self.lattice_cols_label = tk.Label(self.control_frame, text="Lattice Columns:")
        self.lattice_cols_entry = tk.Entry(self.control_frame)
        self.lattice_cols_entry.insert(0, '20')

        self.tolerance_label = tk.Label(self.control_frame, text="Tolerance Threshold:")
        self.tolerance_entry = tk.Entry(self.control_frame)
        self.tolerance_entry.insert(0, '0.5')

        self.empty_ratio_label = tk.Label(self.control_frame, text="Empty Ratio:")
        self.empty_ratio_entry = tk.Entry(self.control_frame)
        self.empty_ratio_entry.insert(0, '0.2')

        self.n_agent_classes_label = tk.Label(self.control_frame, text="Number of Agent Classes:")
        self.n_agent_classes_entry = tk.Entry(self.control_frame)
        self.n_agent_classes_entry.insert(0, '5')

        self.run_button = tk.Button(self.control_frame, text="Run Simulation", command=self.run_simulation)
        self.stop_button = tk.Button(self.control_frame, text="Stop Simulation", command=self.stop_simulation)
        self.reset_button = tk.Button(self.control_frame, text="Reset Simulation", command=self.reset_simulation)

        self.lattice_rows_label.grid(row=0, column=0)
        self.lattice_rows_entry.grid(row=0, column=1)
        self.lattice_cols_label.grid(row=0, column=2)
        self.lattice_cols_entry.grid(row=0, column=3)

        self.tolerance_label.grid(row=1, column=0)
        self.tolerance_entry.grid(row=1, column=1)
        self.empty_ratio_label.grid(row=1, column=2)
        self.empty_ratio_entry.grid(row=1, column=3)

        self.n_agent_classes_label.grid(row=2, column=0)
        self.n_agent_classes_entry.grid(row=2, column=1)
        self.run_button.grid(row=2, column=2)
        self.stop_button.grid(row=2, column=3)
        self.reset_button.grid(row=2, column=4)

        self.generation_label = tk.Label(self.control_frame, text="Generations: 0")
        self.generation_label.grid(row=3, column=0, columnspan=4)

        self.results_button = tk.Button(self.control_frame, text="Analyze Empty Spaces", command=self.analyze_empty_spaces)
        self.results_button.grid(row=4, column=2)

    def analyze_empty_spaces(self):
        empty_ratios = np.arange(0.01, 1.0, 0.01)  # 1% to 99% with increments of 1%
        satisfaction_means = []

        for empty_ratio in empty_ratios:
            # Print the current empty ratio to the terminal
            print(f"Processing Empty Ratio: {empty_ratio:.2f}")  # Print the empty ratio

            # Execute simulation for current empty ratio
            self.reset_simulation(empty_ratio=empty_ratio)
            steps = 1000  # Number of simulation steps for each ratio
            total_satisfaction = 0

            for _ in range(steps):
                self.simulate()
                total_satisfaction += self.model.average_satisfaction()

            average_satisfaction = total_satisfaction / steps
            satisfaction_means.append(average_satisfaction)

        # Plotting results
        plt.figure()
        plt.plot(empty_ratios, satisfaction_means, marker='o')
        plt.title("Average Satisfaction vs Empty Ratio")
        plt.xlabel("Empty Ratio")
        plt.ylabel("Average Satisfaction")
        plt.grid()
        plt.axhline(y=0.5, color='r', linestyle='--')  # Threshold line for satisfaction
        plt.show()


    def run_simulation(self):
        self.reset_simulation()
        self.auto_simulating = True
        self.update_simulation()

    def update_simulation(self):
        if self.auto_simulating:
            moved = self.simulate()
            self.model.draw_graph(self.ax)
            self.canvas.draw()
            self.generation_label.config(text=f"Generations: {self.model.generation_count}")

            # Para garantir que a simulação ocorra por pelo menos mil gerações
            if self.model.generation_count < 1000:
                self.root.after(self.simulation_speed, self.update_simulation)
            else:
                self.stop_simulation()  # Para quando atingir mil gerações

    def stop_simulation(self):
        self.auto_simulating = False

    def reset_simulation(self, empty_ratio=0.2):
        n_agent_classes = int(self.n_agent_classes_entry.get())
        tolerance_threshold = float(self.tolerance_entry.get())
        lattice_rows = int(self.lattice_rows_entry.get())
        lattice_cols = int(self.lattice_cols_entry.get())

        self.model = Schelling(n_agent_classes=n_agent_classes,
                                tolerance_threshold=tolerance_threshold,
                                lattice_m=lattice_rows,
                                lattice_n=lattice_cols,
                                empty_ratio=empty_ratio)

        self.ax.clear()
        self.model.draw_graph(self.ax)
        self.canvas.draw()
        self.generation_label.config(text="Generations: 0")

    def close_program(self):
        self.root.quit()

    def simulate(self):
        moved = self.model.simulate()
        return moved

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = SchellingGUI(root)
    root.mainloop()
