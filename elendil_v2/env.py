import functools
from tabnanny import verbose

# import yaml

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time



class elendil_v2(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "elendil_v2"}

    def __init__(self, 
                 render_mode=None, 
                 num_UGVs=1, 
                 num_UAVs=1, 
                 num_targets=1, 
                 scenario="explore", 
                 map_type="medium",
                 communication_style="none",
                 step_limit=500,
                 seed=None,
                 verbose=False):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.

        -----------------------------------------------------------------------------------------------

        Possible scenarios: 
        - explore: agents explore the map and find targets
        - track: agents track moving targets

        Possible map types:
        - small: 10x10 grid
        - medium: 15x15 grid
        - large: 20x20 grid

        Possible communication styles:
        - none: no communication between agents
        - complete: agents share location of target when found

        State parameters:
        - For each agent:
            - position: (x, y) coordinates on the map
            - vel: (vx, vy) velocity vector
            - fov: (h) h*h square field of view

        - for each target:
            - position: (x, y) coordinates on the map
            - vel: (vx, vy) velocity vector

        - map:
            - size: (width, height)
            - physical_obstacles: list of (x, y) coordinates
            - visual_obstacles: list of (x, y) coordinates

        - goal: (x, y) coordinates of the goal (for explore scenario)

        """
        if seed is not None:
            self._set_seed(seed)
            
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html#numpy.random.Generator.integers
        
        # Definition of parameters

        # self.np_random_seed = seed
        self.verbose = verbose
        self.scenario = scenario
        self.communication_style = communication_style
        self.map = {}
        self.map_type = map_type
        self.map_size = None
        self.step_limit = step_limit
        self.step_count = 0

        self.num_UGVs = num_UGVs
        self.num_UAVs = num_UAVs
        self.num_targets = num_targets

        self.render_mode = render_mode
        self.fig = None
        self.ax = None

        # Running states of the environment used for critic networks
        # For now, let's just store the previous states
        # self.observation_prev = {}
        # self.action_prev = {}
        # self.reward_prev = {}
        # self.termination_prev = {}
        # self.truncation_prev = {}
        # self.info_prev = {}


        # Reward weights
        self.reward_weights = {}
        self.reward_weights["step_penalty"] = -0.01 
        self.reward_weights["obstacle_penalty"] = -0.02
        self.reward_weights["agent_found_target"] = +1
        self.reward_weights["agent_found_by_target"] = -10
        self.reward_weights["target_found_agent"] = +10
        self.reward_weights["target_found_by_agent"] = -1
        self.reward_weights["agent_reached_goal"] = +10
        
        self.state = {}

        # Generation of agent objects
        self.possible_agents = []

        # UAV
        for i in range(self.num_UAVs):
            self._generate_agent("UAV", i)

        if self.verbose: print(f"Generated {self.num_UAVs} UAVs.")

        # UGV
        for j in range(self.num_UGVs):
            self._generate_agent("UGV", j)

        if self.verbose: print(f"Generated {self.num_UGVs} UGVs.")

        # Targets
        for k in range(self.num_targets):
            self._generate_agent("target", k)
        
        if self.verbose: print(f"Generated {self.num_targets} targets.")

        # Define the map:
        self.state["map"] = {} # Map if generated in self.reset()
        
        if self.scenario == "explore":
            # Define goal positions for explore scenario
            self.state["goal_pos"] = np.zeros(2, dtype=int)

        if self.verbose:
            for agent in self.possible_agents:
                print(self.action_space(agent))
                print(self.observation_space(agent))

        self.possible_actions = {
            "UAV": {
                "0": [[0, 1], "move down"],
                "1": [[0, -1], "move up"],
                "2": [[-1, 0], "move left"],
                "3": [[1, 0], "move right"],
                "4": [[0, 0], "do nothing"],
                "5": [[1], "increase altitude"],
                "6": [[0], "hold altitude"],
                "7": [[-1], "decrease altitude"],
            },
            "UGV": {
                "0": [[0, 1], "move down"],
                "1": [[0, -1], "move up"],
                "2": [[-1, 0], "move left"],
                "3": [[1, 0], "move right"],
                "4": [[0, 0], "do nothing"],
            },
            "target": {
                "0": [[0, 1], "move down"],
                "1": [[0, -1], "move up"],
                "2": [[-1, 0], "move left"],
                "3": [[1, 0], "move right"],
                "4": [[0, 0], "do nothing"],
            }
        }

        # Action spaces will be seeded automatically when first accessed via action_space()
        # since they check for np_random_seed in their creation

        self.reset()

        # if self.verbose:
            # print(f"State initialized: {self.state}")

    def _generate_agent(self, agent_type, agent_id):
            agent_name = f"{agent_type}_{agent_id}"
            self.possible_agents.append(agent_name)
            dict = {
                f"{agent_name}_pos": np.zeros(2, dtype=int),
                f"{agent_name}_vel": np.zeros(2, dtype=int),
                f"{agent_name}_fov_dim": np.zeros(1, dtype=int),

            }
            if agent_type == "UAV":
                dict[f"{agent_name}_altitude"] = np.zeros(1, dtype=int)

            if agent_type == "UAV":
                fov_dim = 7
            elif agent_type == "UGV":
                fov_dim = 5
            elif agent_type == "target":
                fov_dim = 3
            dict[f"{agent_name}_fov_dim"] = fov_dim
            dict[f"{agent_name}_fov"] = np.zeros(fov_dim**2, dtype=int)

            self.state.update(dict)
            return 0

    def _generate_map(self):
        '''
        Generates map based on type,
        attributes:
        - size : (width, height)
        - physical_obstacles : list of (x, y) coordinates
        - visual_obstacles : list of (x, y) coordinates
        '''
        if self.map_type == "small":
            self.state["map"]["size"] = (10, 10)
        if self.map_type == "medium":
            self.state["map"]["size"] = (15, 15)
        if self.map_type == "large":
            self.state["map"]["size"] = (20, 20)

        self.state["map"]["physical_obstacles"] = self._generate_obstacles()
        self.state["map"]["visual_obstacles"] = self._generate_obstacles()

        if self.verbose: print(f"Map {self.map_type} loaded.")
            
        return 0
    
    def viz_map(self):
        print_map = np.zeros((self.state["map"]["size"][1], self.state["map"]["size"][0]), dtype=str)
        # Set all free space to " "
        for y in range(self.state["map"]["size"][1]):
            for x in range(self.state["map"]["size"][0]):
                print_map[y, x] = ' '
        # Mark physical obstacles, using a mask
        for i in range(len(self.state["map"]["physical_obstacles"])):
            x, y = self.state["map"]["physical_obstacles"][i]
            print_map[y, x] = 'X'
        # Mark visual obstacles
        for j in range(len(self.state["map"]["visual_obstacles"])):
            x, y = self.state["map"]["visual_obstacles"][j]
            if print_map[y, x] == 'X':
                print_map[y, x] = 'X'  # physical obstacle takes precedence
                continue
            print_map[y, x] = 'v'
        
        # Add agents and targets
        for agent in self.possible_agents:
            x, y = self.state[f"{agent}_pos"]
            print(f"Placing {agent} at ({x}, {y}) on map visualization.")
            if print_map[y, x] == ' ':
                print_map[y, x] = agent[1] if agent.startswith("U") else 'T'  # U for UAV/UGV, T for target

        # Add goal
        x, y = self.state["goal_pos"]
        print_map[y, x] = 'g'
        

        print("Map layout (_: free, X: physical obstacle, h: visual obstacle):")
        print(print_map)
    
    def _generate_obstacles(self):
        # TODO randomly generate obstacles based on map type
        if self.map_type == "small": num_obstacles = 3
        if self.map_type == "medium": num_obstacles = 4
        if self.map_type == "large": num_obstacles = 5

        created = 0
        obstacle_coords = []

        while created < num_obstacles:

            # Random top-left
            ox = int(self.np_random.integers(low = 0, high = self.state["map"]["size"][0], endpoint=False))
            oy = int(self.np_random.integers(low = 0, high = self.state["map"]["size"][1], endpoint=False))

            # Random width and height
            ow = int(self.np_random.integers(low = 1, high = int(self.state["map"]["size"][0] / 3), endpoint=False))
            oh = int(self.np_random.integers(low = 1, high = int(self.state["map"]["size"][1] / 3), endpoint=False))
            
            for x in range(ox, ox + ow):
                for y in range(oy, oy + oh):
                    if x >= self.state["map"]["size"][0] or y >= self.state["map"]["size"][1]:
                        continue
                    obstacle_coords.append([x, y])
            created += 1

        if self.verbose: print(f"Generated obstacle coordinates: {obstacle_coords}")

        return np.array(obstacle_coords)


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memorized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        '''
        Defines the observation space for each agent.
        UAV:
        - [0,1] position (x, y)
        - [2,3] vel (vx, vy)
        - [4] flight level (h)
        - [5] fov (h)
        - [6, 6 + fov**2] vel target (if in fov) (vx, vy)
        - [7 + fov**2, 9] position target (if in fov, or if communication allows) (x, y) * num_targets
        UGV:
        - position (x, y)
        - vel (vx, vy)
        - fov (h)
        - vel target (if in fov) (vx, vy)
        - position target (if in fov, or if communication allows) (x, y) * num_targets
        Target:
        - position (x, y)
        - vel (vx, vy)
        - fov (h)
        - vel agents (if in fov) (vx, vy) * (num_UAVs + num_UGVs)
        - position agents (if in fov) (x, y) * (num_UAVs + num_UGVs)
        '''
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        a = int(np.size(self.state[f"{agent}_pos"]))  # 2
        b = int(np.size(self.state[f"{agent}_vel"]))  # 2
        c = 0
        if agent.startswith("UAV"): 
            c = int(np.size(self.state[f"{agent}_altitude"]))  # 1
        d = int(self.state[f"{agent}_fov_dim"])**2  # fov_dim^2
        e = int(np.size(self.state[f"target_0_vel"])) * self.num_targets  # 2 * num_targets
        f = int(np.size(self.state[f"target_0_pos"])) * self.num_targets  # 2 * num_targets

        g = int(np.size(self.state[f"UAV_0_vel"])) * (self.num_UAVs + self.num_UGVs)  # 2 * (num_UAVs + num_UGVs)
        h = int(np.size(self.state[f"UAV_0_pos"])) * (self.num_UAVs + self.num_UGVs)  # 2 * (num_UAVs + num_UGVs)

        if self.verbose: print(f"Defining observation space for agent: {agent}")
        if agent.startswith("UAV"):
            return Box(low=-10, high=np.inf, shape=(a + b + c + d + e + f,), dtype=np.float32)
        elif agent.startswith("UGV"):
            return Box(low=-10, high=np.inf, shape=(a + b + d + e + f,), dtype=np.float32) 
        elif agent.startswith("target"):
            return Box(low=-10, high=np.inf, shape=(a + b + d + g + h,), dtype=np.float32) 
        else:
            raise ValueError("Unknown agent type: {}".format(agent))

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        print(f"Defining action space for agent: {agent}")
        if agent.startswith("UAV"):
            print(f"-->Action space for {agent} is Discrete(8)")
            space = Discrete(8) # TODO implement UAV action space
                
        elif agent.startswith("UGV"):
            print(f"-->Action space for {agent} is Discrete(5)")
            space = Discrete(5) # TODO implement UGV action space
                
        elif agent.startswith("target"):
            print(f"-->Action space for {agent} is Discrete(5)")
            space = Discrete(5) # TODO implement target action space
        
        else:
            raise ValueError("Unknown agent type: {}".format(agent))
        
        # Seed the action space if we have a seed set
        # Different agents have different seeds to ensure deterministic action sampling is different for each agent
        if hasattr(self, 'np_random_seed'):
            if agent.startswith("UAV_0"):
                alpha = 0
            if agent.startswith("UAV_1"):
                alpha = 1
            if agent.startswith("UAV_2"):
                alpha = 2
            if agent.startswith("UAV_3"):
                alpha = 3
            if agent.startswith("UAV_4"):
                alpha = 4
            if agent.startswith("UAV_5"):
                alpha = 5
            if agent.startswith("UAV_6"):
                alpha = 6

            if agent.startswith("UGV_0"):
                alpha = 7
            if agent.startswith("UGV_1"):
                alpha = 8
            if agent.startswith("UGV_2"):
                alpha = 9
            if agent.startswith("UGV_3"):
                alpha = 10
            if agent.startswith("UGV_4"):
                alpha = 11
            if agent.startswith("UGV_5"):
                alpha = 12
            if agent.startswith("UGV_6"):
                alpha = 13
            
            if agent.startswith("target_0"):
                alpha = 14
            if agent.startswith("target_1"):
                alpha = 15
            if agent.startswith("target_2"):
                alpha = 16
            if agent.startswith("target_3"):
                alpha = 17
            if agent.startswith("target_4"):
                alpha = 18
            if agent.startswith("target_5"):
                alpha = 19
            if agent.startswith("target_6"):
                alpha = 20

            space.seed(self.np_random_seed + alpha)
        
        return space
        

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode == "human":
            self._render_gui()
        else:
            self.viz_map()
    
    def _render_gui(self):
        """
        Simple GUI rendering using matplotlib.
        """
        remove_grid = False

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()  # Turn on interactive mode
        
        self.ax.clear()
        
        # Get map size
        map_width, map_height = self.state["map"]["size"]
        
        # Set up the plot
        self.ax.set_xlim(-0.5, map_width - 0.5)
        self.ax.set_ylim(-0.5, map_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('ELENDIL v2 Environment')
        self.ax.invert_yaxis()  # Match typical grid coordinates
        
        # Draw physical obstacles (red)
        if "physical_obstacles" in self.state["map"]:
            for obs in self.state["map"]["physical_obstacles"]:
                x, y = obs
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                        linewidth=1, edgecolor='black', 
                                        facecolor='lightgray', alpha=0.7,
                                        linestyle='--')
                self.ax.add_patch(rect)
        
        # Draw visual obstacles (yellow)
        if "visual_obstacles" in self.state["map"]:
            for obs in self.state["map"]["visual_obstacles"]:
                x, y = obs
                # Skip if it's also a physical obstacle; outline should be green dashed
                if [x, y] in self.state["map"]["physical_obstacles"].tolist():
                    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                            linewidth=2, edgecolor='green', 
                                            facecolor='lightgray', linestyle='--', alpha=0.5)
                    self.ax.add_patch(rect)

                else: 
                    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor='lightgreen', linestyle='--', alpha=0.5)
                    self.ax.add_patch(rect)
                    

        
        # Draw goal (green star)
        if "goal_pos" in self.state:
            gx, gy = self.state["goal_pos"]
            self.ax.plot(gx, gy, 'g*', markersize=20, label='Goal')
        
        # Draw agents
        for agent in self.possible_agents:
            if agent.startswith("UAV"):
                x, y = self.state[f"{agent}_pos"]
                altitude = self.state.get(f"{agent}_altitude", [0])[0]
                # UAVs shown as blue squares, size based on altitude
                size = 10 + altitude * 5
                self.ax.plot(x, y, f'bs', markersize=size, label='UAV' if agent == self.possible_agents[0] else '')
                self.ax.text(x + 0.3, y + 0.3, agent.split('_')[1], fontsize=8)
                self.ax.text(x + 0.3, y - 0.2, f'{altitude}', fontsize=8)

                # All agents withing the fov of the UAV are shown as dashed boxes
                # Draw the UAV FOV grid using the agent's position and state fov_dim
                fov_dim = self.state[f"{agent}_fov_dim"]
                cx, cy = x, y  # agent center position

                # Draw FOV boxes only for FOV cells not equal to -10
                fov_grid_flat = self.state[f"{agent}_fov"]
                fov_grid = fov_grid_flat.reshape(fov_dim, fov_dim)
                center_idx = fov_dim // 2
                for dx in range(-center_idx, fov_dim - center_idx):
                    for dy in range(-center_idx, fov_dim - center_idx):
                        fx, fy = center_idx + dx, center_idx + dy
                        if 0 <= fx < fov_dim and 0 <= fy < fov_dim:
                            if fov_grid[fx, fy] != -10:
                                px = cx + dx
                                py = cy + dy
                                rect = patches.Rectangle((px - 0.5, py - 0.5), 1, 1, 
                                                        linewidth=1, edgecolor='blue', linestyle='--', alpha=0.1)
                                self.ax.add_patch(rect)
                
            elif agent.startswith("UGV"):
                x, y = self.state[f"{agent}_pos"]
                # UGVs shown as blue circles
                self.ax.plot(x, y, f'bo', markersize=12, label='UGV' if agent == self.possible_agents[0] else '')
                self.ax.text(x + 0.3, y + 0.3, agent.split('_')[1], fontsize=8)
            elif agent.startswith("target"):
                x, y = self.state[f"{agent}_pos"]
                # Targets shown as red triangles
                self.ax.plot(x, y, f'r^', markersize=12, label='Target' if agent == self.possible_agents[0] else '')
                self.ax.text(x + 0.3, y + 0.3, agent.split('_')[1], fontsize=8)
        
        self.ax.legend(loc='upper right')
        if remove_grid:
            self.ax.grid(False)
        else:
            # Configure the grid so that it matches the outer borders of each cell
            grid_size_x, grid_size_y = self.state["map"]["size"]
            self.ax.set_xticks(np.arange(-0.5, grid_size_x, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, grid_size_y, 1), minor=True)
            self.ax.grid(which='minor', color='k', linestyle='-', linewidth=0.6, alpha=0.3)
            self.ax.grid(which='major', alpha=0)  # Hide major gridlines
        plt.draw()
        plt.pause(0.01)  # Small pause to allow the plot to update

    def save_rendering(self, filename=None, dpi=100):
        """
        Saves a rendering of the environment as a PNG file.
        
        Args:
            filename (str, optional): Path to save the PNG file. If None, 
                generates a filename with timestamp.
            dpi (int, optional): Resolution of the saved image. Default is 100.
        
        Returns:
            str: The filename where the image was saved.
        """
        from datetime import datetime
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"elendil_v2_render_{timestamp}.png"
        
        # Ensure filename ends with .png
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Create a temporary figure for saving (non-interactive)
        temp_fig, temp_ax = plt.subplots(figsize=(10, 10))
        plt.ioff()  # Turn off interactive mode for saving
        
        # Get map size
        map_width, map_height = self.state["map"]["size"]
        
        # Set up the plot
        temp_ax.set_xlim(-0.5, map_width - 0.5)
        temp_ax.set_ylim(-0.5, map_height - 0.5)
        temp_ax.set_aspect('equal')
        temp_ax.grid(True, alpha=0.3)
        temp_ax.set_title('ELENDIL v2 Environment')
        temp_ax.invert_yaxis()  # Match typical grid coordinates
        
        # Draw physical obstacles
        if "physical_obstacles" in self.state["map"]:
            for obs in self.state["map"]["physical_obstacles"]:
                x, y = obs
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                        linewidth=1, edgecolor='black', 
                                        facecolor='lightgray', alpha=0.7,
                                        linestyle='--')
                temp_ax.add_patch(rect)
        
        # Draw visual obstacles
        if "visual_obstacles" in self.state["map"]:
            for obs in self.state["map"]["visual_obstacles"]:
                x, y = obs
                # Skip if it's also a physical obstacle; outline should be green dashed
                if [x, y] in self.state["map"]["physical_obstacles"].tolist():
                    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                            linewidth=2, edgecolor='green', 
                                            facecolor='lightgray', linestyle='--', alpha=0.5)
                    temp_ax.add_patch(rect)
                else: 
                    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor='lightgreen', linestyle='--', alpha=0.5)
                    temp_ax.add_patch(rect)
        
        # Draw goal (green star)
        if "goal_pos" in self.state:
            gx, gy = self.state["goal_pos"]
            temp_ax.plot(gx, gy, 'g*', markersize=20, label='Goal')
        
        # Draw agents
        for agent in self.possible_agents:
            if agent.startswith("UAV"):
                x, y = self.state[f"{agent}_pos"]
                altitude = self.state.get(f"{agent}_altitude", [0])[0]
                # UAVs shown as blue squares, size based on altitude
                size = 10 + altitude * 5
                temp_ax.plot(x, y, f'bs', markersize=size, label='UAV' if agent == self.possible_agents[0] else '')
                temp_ax.text(x + 0.3, y + 0.3, agent.split('_')[1], fontsize=8)
                temp_ax.text(x + 0.3, y - 0.2, f'{altitude}', fontsize=8)

                # Draw the UAV FOV grid
                fov_dim = self.state[f"{agent}_fov_dim"]
                cx, cy = x, y  # agent center position

                # Draw FOV boxes only for FOV cells not equal to -10
                fov_grid_flat = self.state[f"{agent}_fov"]
                fov_grid = fov_grid_flat.reshape(fov_dim, fov_dim)
                center_idx = fov_dim // 2
                for dx in range(-center_idx, fov_dim - center_idx):
                    for dy in range(-center_idx, fov_dim - center_idx):
                        fx, fy = center_idx + dx, center_idx + dy
                        if 0 <= fx < fov_dim and 0 <= fy < fov_dim:
                            if fov_grid[fx, fy] != -10:
                                px = cx + dx
                                py = cy + dy
                                rect = patches.Rectangle((px - 0.5, py - 0.5), 1, 1, 
                                                        linewidth=1, edgecolor='blue', linestyle='--', alpha=0.1)
                                temp_ax.add_patch(rect)
                
            elif agent.startswith("UGV"):
                x, y = self.state[f"{agent}_pos"]
                # UGVs shown as blue circles
                temp_ax.plot(x, y, f'bo', markersize=12, label='UGV' if agent == self.possible_agents[0] else '')
                temp_ax.text(x + 0.3, y + 0.3, agent.split('_')[1], fontsize=8)
            elif agent.startswith("target"):
                x, y = self.state[f"{agent}_pos"]
                # Targets shown as red triangles
                temp_ax.plot(x, y, f'r^', markersize=12, label='Target' if agent == self.possible_agents[0] else '')
                temp_ax.text(x + 0.3, y + 0.3, agent.split('_')[1], fontsize=8)
        
        temp_ax.legend(loc='upper right')
        
        # Configure the grid so that it matches the outer borders of each cell
        grid_size_x, grid_size_y = self.state["map"]["size"]
        temp_ax.set_xticks(np.arange(-0.5, grid_size_x, 1), minor=True)
        temp_ax.set_yticks(np.arange(-0.5, grid_size_y, 1), minor=True)
        temp_ax.grid(which='minor', color='k', linestyle='-', linewidth=0.6, alpha=0.3)
        temp_ax.grid(which='major', alpha=0)  # Hide major gridlines
        
        # Save the figure
        temp_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(temp_fig)
        
        # Restore interactive mode if it was on
        if self.render_mode == "human":
            plt.ion()
        
        if self.verbose:
            print(f"Environment rendering saved to: {filename}")
        
        return filename

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        # Destroy env object
        del self

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        if seed is not None:
            self._set_seed(seed)

        self.step_count = 0

        self._generate_map()

        self.agents = self.possible_agents[:] # Reset agents to all possible agents

        for agent in self.agents:
            self._place_agent(agent)

        self._place_agent("goal")

        if self.verbose:
            self.viz_map()

        # Generate observations for each agent
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.num_moves = 0

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        self.step_count += 1

        # User passes an action for each agent
        for agent in self.agents:
            self._update_state(agent, actions.get(agent))

        # Reward based on the new state so R(s')
        rewards = {agent: self._get_reward(agent, actions.get(agent, 0)) for agent in self.agents}

        # Termination based on the new state so T(s')
        terminations = {agent: self._get_termination(agent, actions.get(agent, 0)) for agent in self.agents}
        self.termination_prev = terminations

        # Check if target should be terminated (all other agents are terminated)
        for agent in self.agents:
            if agent.startswith("target"):
                # Check if all UAVs and UGVs are terminated
                all_others_terminated = True
                for other_agent in self.possible_agents:
                    if (other_agent.startswith("UAV") or other_agent.startswith("UGV")):
                        # Check if this agent is still active (not terminated and still in self.agents)
                        if other_agent in self.agents and not terminations.get(other_agent, False):
                            all_others_terminated = False
                            break
                if all_others_terminated:
                    terminations[agent] = True

        # Truncation based on number of steps
        truncations = {agent: self._get_truncation(agent, actions.get(agent, 0)) for agent in self.agents}
    
        # Observations based on the new state so O(s')
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Infos based on the new state so I(s')
        infos = {agent: {self.np_random_seed} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        if self.render_mode == "rgb_array":
            pass # TODO implement rgb_array render mode

        return observations, rewards, terminations, truncations, infos
    

    def _update_state(self, agent, action):
        '''
        Updates the state based on the action of a given agent.
        '''
        # If action is None, do nothing
        if action is None or action == "None":
            return 0

        # I want to represnt all possible actions for each agent in a dictionary
        agent_type = agent.split("_")[0]
        action_str = str(action)

        if self.verbose:
            print(f"Updating state for {agent} with action {action}, {self.possible_actions[agent_type][action_str][0]}")
        
        # Position update (actions 0-4 are movement)
        if int(action) < 5:
            movement = self.possible_actions[agent_type][action_str][0]
            new_x = max(0, min(self.state[f"{agent}_pos"][0] + movement[0], self.state["map"]["size"][0] - 1))
            new_y = max(0, min(self.state[f"{agent}_pos"][1] + movement[1], self.state["map"]["size"][1] - 1))
            if [new_x, new_y] in self.state["map"]["physical_obstacles"].tolist() and agent.startswith("UGV"):
                new_x = self.state[f"{agent}_pos"][0]
                new_y = self.state[f"{agent}_pos"][1]
            else:
                self.state[f"{agent}_pos"][0] = new_x
                self.state[f"{agent}_pos"][1] = new_y
            
        
        # Altitude update (actions 5-7 are altitude changes, only for UAV)
        if agent_type == "UAV" and int(action) >= 5:
            altitude_change = self.possible_actions[agent_type][action_str][0][0]
            new_altitude = self.state[f"{agent}_altitude"][0] + altitude_change
            
            # Clip altitude to valid range (assuming 0-2 or similar)
            self.state[f"{agent}_altitude"][0] = max(0, min(int(new_altitude), 2))

        return 0

    def _place_agent(self, agent_name):
        # Generate random position within map bounds
        x = int(self.np_random.integers(low = 0, high = self.state["map"]["size"][0], endpoint=False))
        y = int(self.np_random.integers(low = 0, high = self.state["map"]["size"][1], endpoint=False))
        # Check for collisions with obstacles and other agents
        if [x, y] not in self.state["map"]["physical_obstacles"].tolist() and all([x, y] not in self.state[f"{agent_name}_pos"].tolist() for agent_name in self.possible_agents):
            self.state[f"{agent_name}_pos"] = np.array([x, y], dtype=int)

        if self.verbose: print(f"Placed {agent_name} at position: {self.state[f'{agent_name}_pos']}")

        return 0
    
    def _set_seed(self, seed=None):
            """
            Sets the seed for this environment's random number generator(s).
            Also seeds all action spaces for deterministic action sampling.
            """
            self.np_random, self.np_random_seed = seeding.np_random(seed)
            
            # Seed all cached action spaces for deterministic sampling
            # This ensures action_space(agent).sample() is deterministic
            # lru_cache will return the same cached space objects
            if hasattr(self, 'possible_agents') and len(self.possible_agents) > 0:
                for agent in self.possible_agents:
                    action_space = self.action_space(agent)
                    action_space.seed(self.np_random_seed)
            
            print(f"Environment seed set to: {self.np_random_seed}")
            return 0
            
    def _get_reward(self, agent, action):
        '''
        Reward logic: ["explore"]
        - UAV & UGV: 
            - -0.01 reward for each step taken
            - -0.02 for bumping into a physical obstacle
            - -1 for being in the field of view of a target (terminates the agent)
            - +1 point for observing the target
            - +10 points for reaching the goal
            
        Target:
            - -0.01point for each step taken
            - -1 point for being in the field of view of a UAV or UGV
            - +10 point for observing the UAV or UGV
        '''

        reward = 0
        agent_pos = self.state[f"{agent}_pos"]
        agent_fov_dim = int(self.state[f"{agent}_fov_dim"])
        agent_half_fov = agent_fov_dim // 2
        
        if agent.startswith("UAV") or agent.startswith("UGV"):
            reward += self.reward_weights["step_penalty"]
            # Check for bumping into physical obstacle
            agent_pos_list = [int(agent_pos[0]), int(agent_pos[1])]
            if agent_pos_list in self.state["map"]["physical_obstacles"].tolist():
                reward += self.reward_weights["obstacle_penalty"]
            
            # Check if reached goal
            if np.array_equal(agent_pos, self.state["goal_pos"]):
                reward += self.reward_weights["agent_reached_goal"]
            
            # Check if agent is in FOV of any target (termination condition)
            ax, ay = int(agent_pos[0]), int(agent_pos[1])
            for k in range(self.num_targets):
                target_name = f"target_{k}"
                target_pos = self.state[f"{target_name}_pos"]
                tx, ty = int(target_pos[0]), int(target_pos[1])
                target_fov_dim = int(self.state[f"{target_name}_fov_dim"])
                target_half_fov = target_fov_dim // 2
                dx = abs(ax - tx)
                dy = abs(ay - ty)
                if dx <= target_half_fov and dy <= target_half_fov:
                    reward += self.reward_weights["agent_found_by_target"]
            
            # Check if observing target (target in agent's FOV)
            for k in range(self.num_targets):
                target_name = f"target_{k}"
                target_pos = self.state[f"{target_name}_pos"]
                tx, ty = int(target_pos[0]), int(target_pos[1])
                dx = abs(ax - tx)
                dy = abs(ay - ty)
                if dx <= agent_half_fov and dy <= agent_half_fov:
                    reward += self.reward_weights["agent_found_target"]
        
        elif agent.startswith("target"):
            reward += self.reward_weights["step_penalty"]
            
            # Check if target is in FOV of any UAV or UGV (penalty)
            tx, ty = int(agent_pos[0]), int(agent_pos[1])
            for other_agent in self.possible_agents:
                if other_agent.startswith("UAV") or other_agent.startswith("UGV"):
                    other_pos = self.state[f"{other_agent}_pos"]
                    ox, oy = int(other_pos[0]), int(other_pos[1])
                    other_fov_dim = int(self.state[f"{other_agent}_fov_dim"])
                    other_half_fov = other_fov_dim // 2
                    dx = abs(tx - ox)
                    dy = abs(ty - oy)
                    if dx <= other_half_fov and dy <= other_half_fov:
                        reward += self.reward_weights["target_found_by_agent"]
            
            # Check if observing UAV or UGV (target sees agent in its FOV)
            for other_agent in self.possible_agents:
                if other_agent.startswith("UAV") or other_agent.startswith("UGV"):
                    other_pos = self.state[f"{other_agent}_pos"]
                    ox, oy = int(other_pos[0]), int(other_pos[1])
                    dx = abs(tx - ox)
                    dy = abs(ty - oy)
                    if dx <= agent_half_fov and dy <= agent_half_fov:
                        reward += self.reward_weights["target_found_agent"]

        return reward

    def _get_observation(self, agent): # TODO implement altitude based masking for the UGV observation
        '''
        Reminder : the observation space for each agent is defined in the observation_space() method.
        UAV:
        - [0,1] position (x, y)
        - [2,3] vel (vx, vy)
        - [4] flight level (h)
        - [5, 5 + fov**2 - 1] fov grid (fov_dim x fov_dim flattened)
        - [5 + fov**2, 5 + fov**2 + 2*num_targets - 1] target velocities (if in fov or communicated)
        - [5 + fov**2 + 2*num_targets, ...] target positions (if in fov or communicated)
        UGV:
        - [0,1] position (x, y)
        - [2,3] vel (vx, vy)
        - [4, 4 + fov**2 - 1] fov grid
        - [4 + fov**2, 4 + fov**2 + 2*num_targets - 1] target velocities
        - [4 + fov**2 + 2*num_targets, ...] target positions
        Target:
        - [0,1] position (x, y)
        - [2,3] vel (vx, vy)
        - [4, 4 + fov**2 - 1] fov grid
        - [4 + fov**2, 4 + fov**2 + 2*(num_UAVs+num_UGVs) - 1] agent velocities
        - [4 + fov**2 + 2*(num_UAVs+num_UGVs), ...] agent positions
        '''
        agent_pos = self.state[f"{agent}_pos"].astype(float)
        agent_vel = self.state[f"{agent}_vel"].astype(float)
        fov_dim = int(self.state[f"{agent}_fov_dim"])
        map_size = self.state["map"]["size"]
        
        # Initialize FOV grid
        # Encoding: -10=out of bounds, 0=free space, 1=obstacle, 2=visual obstacle,
        #           3=ground agent (UGV), 4=air agent (UAV), 5=target, 6=goal
        fov_grid = np.zeros((fov_dim, fov_dim), dtype=np.float32)
        half_fov = fov_dim // 2
        
        # Track what's detected in FOV
        targets_in_fov = set()
        agents_in_fov = set()
        targets_communicated = set()  # For communication style
        
        # Fill FOV grid
        # i is row (y-axis), j is column (x-axis)
        for i in range(fov_dim):
            for j in range(fov_dim):
                # Map FOV grid coordinates to map coordinates
                # j controls x (columns), i controls y (rows)
                map_x = int(agent_pos[0]) + (j - half_fov)
                map_y = int(agent_pos[1]) + (i - half_fov)
                
                # Check if out of bounds
                if map_x < 0 or map_x >= map_size[0] or map_y < 0 or map_y >= map_size[1]:
                    fov_grid[i, j] = 1
                    continue
                
                # Check for physical obstacles
                if [map_x, map_y] in self.state["map"]["physical_obstacles"].tolist():
                    fov_grid[i, j] = 1
                    continue
                
                # Check for visual obstacles
                if [map_x, map_y] in self.state["map"]["visual_obstacles"].tolist():
                    fov_grid[i, j] = 2
                    continue
                
                # Check for goal
                if self.scenario == "explore" and "goal_pos" in self.state:
                    goal_pos = self.state["goal_pos"]
                    if map_x == int(goal_pos[0]) and map_y == int(goal_pos[1]):
                        fov_grid[i, j] = 6
                        continue
                
                # Check for targets at this position
                if agent.startswith("UAV") or agent.startswith("UGV"): # only for UAVs and UGVs
                    for k in range(self.num_targets):
                        target_name = f"target_{k}"
                        target_pos = self.state[f"{target_name}_pos"]
                        if map_x == int(target_pos[0]) and map_y == int(target_pos[1]):
                            if agent.startswith("UAV") and fov_grid[i, j] == 2:
                                continue # visual obstacle takes precedence over target for UAVs
                            fov_grid[i, j] = 5
                            targets_in_fov.add(k)
                            break
                    
                # Check for other agents at this position (only if not already marked)
                if fov_grid[i, j] == 0:
                    for other_agent in self.possible_agents:
                        if other_agent == agent:
                            continue
                        if other_agent.startswith("target"):
                            continue  # Already handled above
                        other_pos = self.state[f"{other_agent}_pos"]
                        if map_x == int(other_pos[0]) and map_y == int(other_pos[1]):
                            # Distinguish between ground and air agents
                            if other_agent.startswith("UAV"):
                                fov_grid[i, j] = 4  # Air agent
                            elif other_agent.startswith("UGV"):
                                fov_grid[i, j] = 3  # Ground agent
                            agents_in_fov.add(other_agent)
                            break
        
        # Handle communication style
        if self.communication_style == "complete":
            # If any agent has detected a target, all agents know about it
            # For now, assume if target is in any agent's FOV, all agents know
            for k in range(self.num_targets):
                target_name = f"target_{k}"
                target_pos = self.state[f"{target_name}_pos"]
                for other_agent in self.possible_agents:
                    if other_agent == agent or other_agent.startswith("target"):
                        continue
                    other_pos = self.state[f"{other_agent}_pos"]
                    other_fov_dim = int(self.state[f"{other_agent}_fov_dim"])
                    other_half_fov = other_fov_dim // 2
                    dx = abs(int(target_pos[0]) - int(other_pos[0]))
                    dy = abs(int(target_pos[1]) - int(other_pos[1]))
                    if dx <= other_half_fov and dy <= other_half_fov:
                        targets_communicated.add(k)
                        break
        
        # Build observation array
        obs_parts = []
        
        # Agent's own position and velocity
        obs_parts.append(agent_pos)
        obs_parts.append(agent_vel)
        
        # Altitude (only for UAV)
        if agent.startswith("UAV"):
            obs_parts.append(np.array([float(self.state[f"{agent}_altitude"][0])], dtype=np.float32))
        
        # Mask the FOV grid based on altitude for UAVs (before flattening)
        if agent.startswith("UAV"):
            altitude = int(self.state[f"{agent}_altitude"][0])
            if altitude == 2:
                # Full FOV - no masking
                pass
            elif altitude == 1:
                # Mask outer layer: set first and last row/column to -10
                fov_grid[0, :] = -10  # First row
                fov_grid[-1, :] = -10  # Last row
                fov_grid[:, 0] = -10  # First column
                fov_grid[:, -1] = -10  # Last column
            elif altitude == 0:
                # Mask outer 2 layers
                fov_grid[0:2, :] = -10  # First 2 rows
                fov_grid[-2:, :] = -10  # Last 2 rows
                fov_grid[:, 0:2] = -10  # First 2 columns
                fov_grid[:, -2:] = -10  # Last 2 columns

        # FOV grid (flattened) - add after masking
        obs_parts.append(fov_grid.flatten())

        # Update the global state of the environment with the new FOV grid
        self.state[f"{agent}_fov"] = fov_grid.flatten()
        
        if agent.startswith("UAV") or agent.startswith("UGV"):
            # Target velocities and positions
            target_vels = np.zeros(2 * self.num_targets, dtype=np.float32)
            target_positions = np.zeros(2 * self.num_targets, dtype=np.float32)
            
            for k in range(self.num_targets):
                target_name = f"target_{k}"
                idx_start = k * 2
                idx_end = (k + 1) * 2
                
                # Check if target is in FOV or communicated
                if k in targets_in_fov or k in targets_communicated:
                    target_vel = self.state[f"{target_name}_vel"].astype(float)
                    target_pos = self.state[f"{target_name}_pos"].astype(float)
                    target_vels[idx_start:idx_end] = target_vel
                    target_positions[idx_start:idx_end] = target_pos
                else:
                    # Not visible - use placeholder values (0 or -10)
                    target_vels[idx_start:idx_end] = -10
                    target_positions[idx_start:idx_end] = -10
            
            obs_parts.append(target_vels)
            obs_parts.append(target_positions)
            
        elif agent.startswith("target"):
            # Agent velocities and positions
            agent_vels = np.zeros(2 * (self.num_UAVs + self.num_UGVs), dtype=np.float32)
            agent_positions = np.zeros(2 * (self.num_UAVs + self.num_UGVs), dtype=np.float32)
            
            agent_idx = 0
            for i in range(self.num_UAVs):
                uav_name = f"UAV_{i}"
                idx_start = agent_idx * 2
                idx_end = (agent_idx + 1) * 2
                
                if uav_name in agents_in_fov:
                    uav_vel = self.state[f"{uav_name}_vel"].astype(float)
                    uav_pos = self.state[f"{uav_name}_pos"].astype(float)
                    agent_vels[idx_start:idx_end] = uav_vel
                    agent_positions[idx_start:idx_end] = uav_pos
                else:
                    agent_vels[idx_start:idx_end] = -10
                    agent_positions[idx_start:idx_end] = -10
                
                agent_idx += 1
            
            for j in range(self.num_UGVs):
                ugv_name = f"UGV_{j}"
                idx_start = agent_idx * 2
                idx_end = (agent_idx + 1) * 2
                
                if ugv_name in agents_in_fov:
                    ugv_vel = self.state[f"{ugv_name}_vel"].astype(float)
                    ugv_pos = self.state[f"{ugv_name}_pos"].astype(float)
                    agent_vels[idx_start:idx_end] = ugv_vel
                    agent_positions[idx_start:idx_end] = ugv_pos
                else:
                    agent_vels[idx_start:idx_end] = -10
                    agent_positions[idx_start:idx_end] = -10
                
                agent_idx += 1
            
            obs_parts.append(agent_vels)
            obs_parts.append(agent_positions)
        
        # Concatenate all parts
        observation = np.concatenate(obs_parts, dtype=np.float32)
        
        return observation

    def _get_termination(self, agent, action):
        '''
        Termination logic:
        UAV :
        - Terminated if UAV enters fov of a target
        - Terminated if UAV reaches goal
        UGV:
        - Terminated if UGV enters fov of a target
        - Terminated if UGV reaches goal
        Target:
        - TODO implement target termination function
        '''

        if agent.startswith("UAV") or agent.startswith("UGV"):
            # Check if agent reached goal
            agent_pos = self.state[f"{agent}_pos"]
            goal_pos = self.state["goal_pos"]
            if np.array_equal(agent_pos, goal_pos):
                return True

            # Check if agent is in FOV of any target
            # Simple bounds check: agent is within square FOV of target
            # FOV is a square of size fov_dim x fov_dim centered on the target
            ax, ay = int(agent_pos[0]), int(agent_pos[1])
            for k in range(self.num_targets):
                target_name = f"target_{k}"
                target_pos = self.state[f"{target_name}_pos"]
                tx, ty = int(target_pos[0]), int(target_pos[1])
                target_fov_dim = int(self.state[f"{target_name}_fov_dim"])
                half_fov = target_fov_dim // 2
                
                # Check if agent is within square FOV bounds
                # For fov_dim=3, half_fov=1, covers [tx-1, tx+1] and [ty-1, ty+1]
                dx = abs(ax - tx)
                dy = abs(ay - ty)
                
                if dx <= half_fov and dy <= half_fov:
                    return True
        
        elif agent.startswith("target"):
            # The target is only terminated if all other agents are terminated
            return False
        
        return False

    def _get_truncation(self, agent, action):
        # The environment is truncated if the number of steps exceeds the step limit
        return self.step_count >= self.step_limit

    def viz_fov_grid(self, agent):
        if agent.startswith("UAV"):
            fov_grid = self._observation_function(agent)[5:5 + int(self.state[f"{agent}_fov_dim"])**2]
            fov_grid = fov_grid.reshape(int(self.state[f"{agent}_fov_dim"]), int(self.state[f"{agent}_fov_dim"]))
            print(f"FOV grid for {agent}: \n {fov_grid}")
            return 0
        elif agent.startswith("UGV"):
            fov_grid = self._observation_function(agent)[4:4 + int(self.state[f"{agent}_fov_dim"])**2]
            fov_grid = fov_grid.reshape(int(self.state[f"{agent}_fov_dim"]), int(self.state[f"{agent}_fov_dim"]))
            print(f"FOV grid for {agent}: \n {fov_grid}")
            return 0
        elif agent.startswith("target"):
            fov_grid = self._observation_function(agent)[4:4 + int(self.state[f"{agent}_fov_dim"])**2]
            fov_grid = fov_grid.reshape(int(self.state[f"{agent}_fov_dim"]), int(self.state[f"{agent}_fov_dim"]))
            print(f"FOV grid for {agent}: \n {fov_grid}")
            return 0
        else:
            raise ValueError("Unknown agent type: {}".format(agent))

    def viz_observation(self, agent):
        """
        Visualizes the observation for a given agent in a human-readable format.
        """
        obs = self._get_observation(agent)
        fov_dim = int(self.state[f"{agent}_fov_dim"])
        
        print("\n" + "="*70)
        print(f"OBSERVATION FOR {agent}")
        print("="*70)
        
        # Extract indices
        idx = 0
        
        # Position
        pos = obs[idx:idx+2]
        print(f"\nPosition (x, y):")
        print(f"   [{pos[0]:.1f}, {pos[1]:.1f}]")
        idx += 2
        
        # Velocity
        vel = obs[idx:idx+2]
        print(f"\nVelocity (vx, vy):")
        print(f"   [{vel[0]:.1f}, {vel[1]:.1f}]")
        idx += 2
        
        # Altitude (only for UAV)
        if agent.startswith("UAV"):
            altitude = obs[idx:idx+1]
            print(f"\nAltitude:")
            print(f"   [{altitude[0]:.1f}]")
            idx += 1
        
        # FOV Grid
        fov_size = fov_dim ** 2
        fov_grid = obs[idx:idx+fov_size].reshape(fov_dim, fov_dim)
        print(f"\nField of View (FOV) Grid ({fov_dim}x{fov_dim}):")
        print(f"   Legend: .=free space, X=obstacle, h=visual obstacle, G=ground agent (UGV), A=air agent (UAV), T=target, g=goal, #=out of bounds")
        # Print with border for better visualization
        print("   " + "-" * (fov_dim * 4 + 1))
        for i in range(fov_dim):
            row_str = "   |"
            for j in range(fov_dim):
                val = int(fov_grid[i, j])
                if val == 0:
                    row_str += " . "
                elif val == 1:
                    row_str += " X "  # Physical obstacle or out of bounds
                elif val == 2:
                    row_str += " v "  # Visual obstacle
                elif val == 3:
                    row_str += " G "  # Ground agent (UGV)
                elif val == 4:
                    row_str += " A "  # Air agent (UAV)
                elif val == 5:
                    row_str += " T "  # Target
                elif val == 6:
                    row_str += " g "  # Goal
                elif val == -10:
                    row_str += " # "  # Limited visibility of UAV
                else:
                    row_str += f"{val:^3}"
                row_str += "|"
            print(row_str)
        print("   " + "-" * (fov_dim * 4 + 1))
        idx += fov_size
        
        if agent.startswith("UAV") or agent.startswith("UGV"):
            # Target velocities
            target_vels = obs[idx:idx + 2*self.num_targets]
            print(f"\nTarget Velocities:")
            for k in range(self.num_targets):
                start_idx = k * 2
                end_idx = (k + 1) * 2
                vel = target_vels[start_idx:end_idx]
                if vel[0] == -10 and vel[1] == -10:
                    print(f"   Target {k}: Not visible")
                else:
                    print(f"   Target {k}: [{vel[0]:.1f}, {vel[1]:.1f}]")
            idx += 2*self.num_targets
            
            # Target positions
            target_positions = obs[idx:idx + 2*self.num_targets]
            print(f"\nTarget Positions (relative):")
            for k in range(self.num_targets):
                start_idx = k * 2
                end_idx = (k + 1) * 2
                pos = target_positions[start_idx:end_idx]
                if pos[0] == -10 and pos[1] == -10:
                    print(f"   Target {k}: Not visible")
                else:
                    print(f"   Target {k}: [{pos[0]:.1f}, {pos[1]:.1f}]")
            idx += 2*self.num_targets
            
        elif agent.startswith("target"):
            # Agent velocities
            agent_vels = obs[idx:idx + 2*(self.num_UAVs + self.num_UGVs)]
            print(f"\nAgent Velocities:")
            agent_idx = 0
            for i in range(self.num_UAVs):
                uav_name = f"UAV_{i}"
                start_idx = agent_idx * 2
                end_idx = (agent_idx + 1) * 2
                vel = agent_vels[start_idx:end_idx]
                if vel[0] == -10 and vel[1] == -10:
                    print(f"   {uav_name}: Not visible")
                else:
                    print(f"   {uav_name}: [{vel[0]:.1f}, {vel[1]:.1f}]")
                agent_idx += 1
            
            for j in range(self.num_UGVs):
                ugv_name = f"UGV_{j}"
                start_idx = agent_idx * 2
                end_idx = (agent_idx + 1) * 2
                vel = agent_vels[start_idx:end_idx]
                if vel[0] == -10 and vel[1] == -10:
                    print(f"   {ugv_name}: Not visible")
                else:
                    print(f"   {ugv_name}: [{vel[0]:.1f}, {vel[1]:.1f}]")
                agent_idx += 1
            idx += 2*(self.num_UAVs + self.num_UGVs)
            
            # Agent positions
            agent_positions = obs[idx:idx + 2*(self.num_UAVs + self.num_UGVs)]
            print(f"\nAgent Positions (relative):")
            agent_idx = 0
            for i in range(self.num_UAVs):
                uav_name = f"UAV_{i}"
                start_idx = agent_idx * 2
                end_idx = (agent_idx + 1) * 2
                pos = agent_positions[start_idx:end_idx]
                if pos[0] == -10 and pos[1] == -10:
                    print(f"   {uav_name}: Not visible")
                else:
                    print(f"   {uav_name}: [{pos[0]:.1f}, {pos[1]:.1f}]")
                agent_idx += 1
            
            for j in range(self.num_UGVs):
                ugv_name = f"UGV_{j}"
                start_idx = agent_idx * 2
                end_idx = (agent_idx + 1) * 2
                pos = agent_positions[start_idx:end_idx]
                if pos[0] == -10 and pos[1] == -10:
                    print(f"   {ugv_name}: Not visible")
                else:
                    print(f"   {ugv_name}: [{pos[0]:.1f}, {pos[1]:.1f}]")
                agent_idx += 1
            idx += 2*(self.num_UAVs + self.num_UGVs)
        
        print("\n" + "="*70)
        print(f"Total observation size: {len(obs)}")
        print("="*70 + "\n")
        
        return 0
    


if __name__ == "__main__":
    env = elendil_v2(
        render_mode="human",
        num_UGVs=1,
        num_UAVs=1,
        num_targets=1,
        scenario="explore",
        map_type="medium",
        step_limit=500,
        seed=43,
        verbose=True,
    )

    # Save rendering to file
    # env.save_rendering("elendil_v2.png")
    num_episodes = 1  # or whatever
    for ep in range(num_episodes):
        observations, infos = env.reset()

        # initialize done flags
        terminations = {agent: False for agent in env.agents}
        truncations = {agent: False for agent in env.agents}

        # run until everyone is done OR truncated
        while not all(terminations[a] or truncations[a] for a in env.agents):
            # only act for still-alive agents
            alive_agents = [
                a for a in env.agents
                if not (terminations[a] or truncations[a])
            ]

            actions = {
                agent: env.action_space(agent).sample()
                for agent in alive_agents
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)

            env.viz_map()
            # pick any agent that still exists (may be done, but fine for debugging)
            env.viz_observation(env.agents[0])
            time.sleep(0.2)

    env.close()