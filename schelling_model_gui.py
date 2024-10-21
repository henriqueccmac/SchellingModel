import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

DEBUG = False 
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

    def get_neighbors(self, node):
        """Returns a list of all neighbors of node, ie all nodes that have an edge connected to it"""
        return list(self.graph.neighbors(node))
    
    def is_unsatisfied(self, node):
        """Checks if neighbors are at least node.treshold similar to node"""

        agent_id = self.graph.nodes[node]['class']
        if agent_id is None:
            return False

        neighborhood = self.get_neighbors(node)
        similar_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] == agent_id)
        occupied_count = sum(1 for neighbor in neighborhood if self.graph.nodes[neighbor]['class'] is not None)

        if occupied_count == 0:  # No neighbors
            return False

        satisfaction_ratio = similar_count / occupied_count

        if DEBUG:
            print(f"Node at {node} (Class: {agent_id}) - Satisfaction ratio: {satisfaction_ratio}, Threshold: {self.graph.nodes[node]['threshold']}")

        # The agent is unsatisfied if the similarity ratio is below its threshold
        return satisfaction_ratio < self.graph.nodes[node]['threshold']
    
    def move_agent(self, node):
        """Move agent to any free position/node in neighborhood"""

        neighbor_positions = self.get_neighbors(node)
        available_positions = [n for n in neighbor_positions if self.graph.nodes[n]['class'] is None]

        if not available_positions:
            return

        new_position = tuple(self.random_seeded.choice(available_positions))

        self.graph.nodes[new_position]['class'] = self.graph.nodes[tuple(node)]['class']
        self.graph.nodes[tuple(node)]['class'] = None

    def simulate(self):
        """Move agents according to their satisfaction"""

        all_nodes = list(self.graph.nodes)
        self.random_seeded.shuffle(all_nodes) 
        
        unsatisfied_agents = [node for node in all_nodes if self.is_unsatisfied(node)]

        if not unsatisfied_agents:
            return False # All satisfied, stop simulation
        
        for node in unsatisfied_agents:
            self.move_agent(node)
        return True
    
    def draw_graph(self, ax):
        """Draw the grid using uniformly sized squares that touch each other."""
        
        # Create a matrix to represent the grid
        grid = np.zeros((self.lattice_m, self.lattice_n))

        # Fill the grid matrix with agent classes
        for (x, y), data in self.graph.nodes(data=True):
            agent_class = data.get('class')
            grid[x, y] = agent_class if agent_class is not None else 0  # Empty spaces get a 0

        # Create a color map for the agents
        colors = ['#FFFFFF', '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3']  # White for empty and colors for classes
        cmap = plt.matplotlib.colors.ListedColormap(colors[:self.n_agent_classes + 1])

        ax.clear()  # Clear the axis

        # Use imshow to display the grid as uniformly sized squares
        ax.imshow(grid, cmap=cmap, origin='upper', extent=[0, self.lattice_n, 0, self.lattice_m])

        # Remove the grid and axis labels to make it look clean
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')  # Ensure squares remain square-shaped

        # Hide the axes spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)



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
# GUI Class for the Schelling Model
class SchellingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Schelling Segregation Model")
        self.auto_simulating = False
        self.simulation_speed = 400

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

        #Close program
        self.root.protocol("WM_DELETE_WINDOW", self.close_program)

        # Initialize Schelling model with default values
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


            # Speed control
        self.decrease_speed_button = tk.Button(self.control_frame, text="-", command=self.decrease_speed)
        self.speed_label = tk.Label(self.control_frame, text=f"Speed: {self.simulation_speed} ms")
        self.increase_speed_button = tk.Button(self.control_frame, text="+", command=self.increase_speed)

        # Place speed control elements aligned to the left
        self.decrease_speed_button.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.speed_label.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.increase_speed_button.grid(row=3, column=2, padx=5, pady=5, sticky='w')

        
        # Fixed layout for controls
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

    def toggle_auto_simulation(self):
        self.auto_simulating = not self.auto_simulating
        if self.auto_simulating:
            self.auto_button.config(text="Stop Auto")
            self.run_auto_simulation()
        else:
            self.auto_button.config(text="Start Auto")

    def run_auto_simulation(self):
        if self.auto_simulating:
            moved = self.model.simulate()  # Simulate and check if anything moved
            if moved:
                self.generation += 1  # Increment generation
                self.update_plot()
                self.root.after(self.simulation_speed, self.run_auto_simulation)  # Continue auto-simulation
            else:
                self.auto_simulating = False
                self.auto_button.config(text="Start Auto")  # Stop if no movement

    def increase_speed(self):
        if self.simulation_speed > 50:  
            self.simulation_speed -= 50
        self.update_speed_label()

    def decrease_speed(self):
        self.simulation_speed += 50  # Increase delay (decrease speed)
        self.update_speed_label()

    def update_speed_label(self):
        self.speed_label.config(text=f"Speed: {self.simulation_speed} ms")

    def step_simulation(self):
        moved = self.model.simulate()  # Get whether agents moved
        if moved:
            self.generation += 1  # Increment generation only if agents moved
            self.update_plot()
        else:
            self.auto_simulating = False
            self.auto_button.config(text="Start Auto")  # Stop auto-simulation if no agents moved


    def reset_simulation(self):
        # Stop auto-simulation if it is running
        self.auto_simulating = False
        self.auto_button.config(text="Start Auto")  # Reset the auto button label
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
        self.generation = 1  # Reset generation to 1
        self.update_plot()

    def update_plot(self):
        self.model.draw_graph(self.ax)
        self.ax.set_title(f"Generation: {self.generation}")
        self.canvas.draw()

    def close_program(self):
        # Stop auto-simulation if it's running
        self.auto_simulating = False
        
        # Perform any necessary cleanup here if needed
        print("Closing the application...")

        # Exit the Tkinter main loop and close the program
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SchellingGUI(root)
    root.mainloop()


