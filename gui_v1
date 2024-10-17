import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Schelling Model Class with Diagonal Neighbors
class Schelling:
    def __init__(self, n_agent_classes, tolerance_treshold, lattice_m, lattice_n, empty_ratio, seed=0):
        self.seed = seed
        self.empty_ratio = empty_ratio
        self.tolerance_treshold = tolerance_treshold
        self.n_agent_classes = n_agent_classes
        
        self.lattice_m = lattice_m
        self.lattice_n = lattice_n
        self.graph = nx.grid_2d_graph(lattice_m, lattice_n)
        self.graph = self.add_diagonal_edges(self.graph)

        self.population_distribution = np.full(n_agent_classes + 1, (1 - empty_ratio) / n_agent_classes)
        self.population_distribution[0] = empty_ratio  # Empty spaces

        self.random_seeded = np.random.default_rng(seed)
        self.assign_agents()

    def add_diagonal_edges(self, graph):
        """Add diagonal edges to existing grid graph"""
        for x in range(self.lattice_m):
            for y in range(self.lattice_n):
                node = (x, y)
                diagonal_neighbors = {(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)}
                for n in diagonal_neighbors:
                    if n in graph.nodes:
                        graph.add_edge(node, n)
        return graph

    def assign_agents(self):
        """Assign agents to the graph according to empty ratio and number of agent classes"""
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
            self.graph.nodes[pos]['threshold'] = self.tolerance_treshold
            self.graph.nodes[pos]['movable'] = True

    def get_neighbors(self, node):
        """Returns a list of neighbors of node, including diagonal ones"""
        return list(self.graph.neighbors(node))
    
    def is_unsatisfied(self, node):
        """Checks if the agent is unsatisfied based on neighbors' similarity"""
        if self.graph.nodes[node]['movable'] == False:
            return False  # Can't move, no point in checking satisfaction

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
                if neighbor_agent_id == agent_id:
                    similar_count += 1

        if occupied_count == 0:
            return False

        satisfaction_ratio = similar_count / occupied_count
        print(f"Node at {node} (Class: {agent_id}) - Satisfaction ratio: {satisfaction_ratio}, Threshold: {self.graph.nodes[node]['threshold']}")
        return satisfaction_ratio < self.graph.nodes[node]['threshold']
    
    def move_agent(self, node):
        """Move agent to any free neighboring position if possible"""
        if self.graph.nodes[node]['movable'] == False:
            return

        neighbor_positions = self.get_neighbors(node)
        available_positions = [n for n in neighbor_positions if self.graph.nodes[n]['class'] is None]

        if not available_positions:
            self.graph.nodes[node]['movable'] = False  # Surrounded, can't move
            return

        new_position = tuple(self.random_seeded.choice(available_positions))
        self.graph.nodes[new_position]['class'] = self.graph.nodes[node]['class']
        self.graph.nodes[node]['class'] = None

    def is_balanced(self):
        """Check if all agents are satisfied"""
        for node in self.graph.nodes:
            if self.graph.nodes[node]['class'] is not None:
                if self.is_unsatisfied(node):
                    return False
        return True

    def simulate(self):
        """Simulate one step of the model"""
        for node in self.graph.nodes:
            if self.graph.nodes[node]['class'] is not None:
                if self.is_unsatisfied(node):
                    self.move_agent(node)

    def draw_graph(self, ax):
        """Draw the current state of the grid"""
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3']
        agent_color_map = {i + 1: colors[i % len(colors)] for i in range(self.n_agent_classes)}
        agent_color_map[None] = '#FFFFFF'

        node_color = [agent_color_map.get(self.graph.nodes[node]['class'], '#FFFFFF') for node in self.graph.nodes()]
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

# GUI Class for the Schelling Model
class SchellingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Schelling Segregation Model")
        self.auto_simulating = False
        self.generation = 1  # Start generation at 1

        # Create a frame for the plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.grid(row=0, column=0, columnspan=4, sticky='nsew')

        # Create a frame for the controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=4, sticky='ew')

        # Set resizing rules
        self.root.grid_rowconfigure(0, weight=4)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        # Initialize Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create parameter input fields
        self.setup_widgets()

        # Initialize Schelling model with default values
        self.reset_simulation()

        # Bind the resize event to update the layout dynamically
        self.root.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        window_width = event.width
        if window_width < 500:
            self.arrange_controls(2)  # 2 columns in minimized version
        else:
            self.arrange_controls(4)  # 1 row in maximized version

    def arrange_controls(self, columns):
        for widget in self.control_frame.winfo_children():
            widget.grid_forget()

        if columns == 2:
            self.lattice_rows_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.lattice_rows_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

            self.lattice_cols_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
            self.lattice_cols_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

            self.tolerance_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.tolerance_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

            self.num_agents_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
            self.num_agents_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')

            self.empty_ratio_label.grid(row=4, column=0, padx=5, pady=5, sticky='w')
            self.empty_ratio_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')

            self.step_button.grid(row=5, column=0)
            self.auto_button.grid(row=5, column=1)
            self.reset_button.grid(row=5, column=2)

        elif columns == 4:
            self.lattice_rows_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.lattice_rows_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
            
            self.lattice_cols_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
            self.lattice_cols_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')

            self.tolerance_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
            self.tolerance_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

            self.num_agents_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')
            self.num_agents_entry.grid(row=1, column=3, padx=5, pady=5, sticky='w')

            self.empty_ratio_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.empty_ratio_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

            self.step_button.grid(row=2, column=2)
            self.auto_button.grid(row=2, column=3)
            self.reset_button.grid(row=2, column=4)

    def setup_widgets(self):
        self.lattice_rows_label = tk.Label(self.control_frame, text="Lattice Rows:")
        self.lattice_rows_entry = tk.Entry(self.control_frame)
        self.lattice_rows_entry.insert(0, '20')

        self.lattice_cols_label = tk.Label(self.control_frame, text="Lattice Columns:")
        self.lattice_cols_entry = tk.Entry(self.control_frame)
        self.lattice_cols_entry.insert(0, '20')

        self.tolerance_label = tk.Label(self.control_frame, text="Tolerance Threshold:")
        self.tolerance_entry = tk.Entry(self.control_frame)
        self.tolerance_entry.insert(0, '0.3')

        self.num_agents_label = tk.Label(self.control_frame, text="Number of Agent Classes:")
        self.num_agents_entry = tk.Entry(self.control_frame)
        self.num_agents_entry.insert(0, '3')

        self.empty_ratio_label = tk.Label(self.control_frame, text="Empty Ratio:")
        self.empty_ratio_entry = tk.Entry(self.control_frame)
        self.empty_ratio_entry.insert(0, '0.1')

        self.step_button = tk.Button(self.control_frame, text="Step", command=self.step_simulation)
        self.auto_button = tk.Button(self.control_frame, text="Start Auto", command=self.toggle_auto_simulation)
        self.reset_button = tk.Button(self.control_frame, text="Reset", command=self.reset_simulation)

    def toggle_auto_simulation(self):
        self.auto_simulating = not self.auto_simulating
        if self.auto_simulating:
            self.auto_button.config(text="Stop Auto")
            self.run_auto_simulation()
        else:
            self.auto_button.config(text="Start Auto")

    def run_auto_simulation(self):
        if self.auto_simulating:
            self.step_simulation()
            self.root.after(500, self.run_auto_simulation)

    def step_simulation(self):
        self.model.simulate()
        self.generation += 1
        self.update_plot()

    def reset_simulation(self):
        lattice_rows = int(self.lattice_rows_entry.get())
        lattice_cols = int(self.lattice_cols_entry.get())
        tolerance = float(self.tolerance_entry.get())
        num_agents = int(self.num_agents_entry.get())
        empty_ratio = float(self.empty_ratio_entry.get())

        self.model = Schelling(
            n_agent_classes=num_agents,
            tolerance_treshold=tolerance,
            lattice_m=lattice_rows,
            lattice_n=lattice_cols,
            empty_ratio=empty_ratio
        )
        self.generation = 1
        self.update_plot()

    def update_plot(self):
        self.model.draw_graph(self.ax)
        self.ax.set_title(f"Generation: {self.generation}")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SchellingGUI(root)
    root.mainloop()
