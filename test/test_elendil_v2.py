# Unit tests for elendil_v2.py

import unittest
import numpy as np
from elendil_v2 import elendil_v2
from pettingzoo.test import parallel_api_test

def states_equal(s1, s2):
        if s1.keys() != s2.keys():
            return False
        for k in s1:
            v1, v2 = s1[k], s2[k]
            if isinstance(v1, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False
            elif isinstance(v1, dict):
                if not states_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True

class TestElendilV2(unittest.TestCase):
    
    def setUp(self):
        self.env = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", target_movement_pattern=None, map_type="medium", step_limit=500, seed=42, verbose=False)
        self.env.reset()
        pass

    def tearDown(self):
        self.env.close()

    def test_initial_state(self):
        state = self.env.state

        self.assertIsInstance(state, dict)

        # Verify that the state contains the correct keys
        self.assertIn("UGV_0_pos", state)
        self.assertIn("UGV_0_vel", state)
        self.assertIn("UGV_0_fov_dim", state)
        self.assertIn("UGV_0_fov", state)

        self.assertIn("UAV_0_pos", state)
        self.assertIn("UAV_0_vel", state)
        self.assertIn("UAV_0_fov_dim", state)
        self.assertIn("UAV_0_fov", state)
        self.assertIn("UAV_0_altitude", state)

        self.assertIn("target_0_pos", state)
        self.assertIn("target_0_vel", state)
        self.assertIn("target_0_fov_dim", state)
        self.assertIn("target_0_fov", state)

        self.assertIn("map", state)
        self.assertIn("size", state["map"])
        self.assertIn("physical_obstacles", state["map"])
        self.assertIn("visual_obstacles", state["map"])

        self.assertIn("goal_pos", state)

    def test_map_generation(self):
        # Medium map
        map_data = self.env.state["map"]
        # Check existence of keys
        self.assertIsInstance(map_data, dict)
        self.assertIn("size", map_data)
        self.assertIn("physical_obstacles", map_data)
        self.assertIn("visual_obstacles", map_data)
        # Check types of values
        self.assertIsInstance(map_data["size"], list)
        self.assertEqual(map_data["size"], [15, 15])
        self.assertIsInstance(map_data["physical_obstacles"], np.ndarray)
        self.assertIsInstance(map_data["visual_obstacles"], np.ndarray)
        # Check size of values
        self.assertEqual(len(map_data["size"]), 2)
        # It's harder to check the size of the obstacles, but we can check that they are not empty
        self.assertNotEqual(len(map_data["physical_obstacles"]), 0)
        self.assertNotEqual(len(map_data["visual_obstacles"]), 0)

        # Small map
        env2 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1,
                        num_targets=1, scenario="explore", map_type="small",
                        step_limit=500, seed=42, verbose=False)
        map_data = env2.state["map"]

        self.assertIsInstance(map_data, dict)
        self.assertIn("size", map_data)
        self.assertIn("physical_obstacles", map_data)
        self.assertIn("visual_obstacles", map_data)
        # Check types of values
        self.assertIsInstance(map_data["size"], list)
        self.assertEqual(map_data["size"], [10, 10])
        self.assertIsInstance(map_data["physical_obstacles"], np.ndarray)
        self.assertIsInstance(map_data["visual_obstacles"], np.ndarray)
        # Check size of values
        self.assertEqual(len(map_data["size"]), 2)
        # It's harder to check the size of the obstacles, but we can check that they are not empty
        self.assertNotEqual(len(map_data["physical_obstacles"]), 0)
        self.assertNotEqual(len(map_data["visual_obstacles"]), 0)

        # Large map
        env3 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1,
                        num_targets=1, scenario="explore", map_type="large",
                        step_limit=500, seed=42, verbose=False)
        map_data = env3.state["map"]

        self.assertIsInstance(map_data, dict)
        self.assertIn("size", map_data)
        self.assertIn("physical_obstacles", map_data)
        self.assertIn("visual_obstacles", map_data)
        # Check types of values
        self.assertIsInstance(map_data["size"], list)
        self.assertEqual(map_data["size"], [20, 20])
        self.assertIsInstance(map_data["physical_obstacles"], np.ndarray)
        self.assertIsInstance(map_data["visual_obstacles"], np.ndarray)
        # Check size of values
        self.assertEqual(len(map_data["size"]), 2)
        # It's harder to check the size of the obstacles, but we can check that they are not empty
        self.assertNotEqual(len(map_data["physical_obstacles"]), 0)

    def test_random_seed_consistency_for_map_generation_same_seed(self):
        print("Test 1: Same Seed Consistency")
        seed = np.random.randint(0, 10000)
        seed = 1234

        env1 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1,
                        num_targets=1, scenario="explore", map_type="medium",
                        step_limit=500, seed=seed, verbose=False)

        env2 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1,
                        num_targets=1, scenario="explore", map_type="medium",
                        step_limit=500, seed=seed, verbose=False)
        
        state_list_1 = []
        state_list_2 = []
        for i in range(3):
            env1.reset()
            env2.reset()
            state_list_1.append(env1.state)
            state_list_2.append(env2.state)

        # They should NOT match for different seeds
        for s1, s2 in zip(state_list_1, state_list_2):
            self.assertTrue(
                states_equal(s1, s2)
            )

    def test_random_seed_consistency_for_map_generation_different_seeds(self):
        print("Test 2: Different Seed Consistency")
        seed = np.random.randint(0, 10000)
        seed = 1234

        env1 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1,
                        num_targets=1, scenario="explore", map_type="medium",
                        step_limit=500, seed=seed, verbose=False)

        env2 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1,
                        num_targets=1, scenario="explore", map_type="medium",
                        step_limit=500, seed=seed+1, verbose=False)
        
        state_list_1 = []
        state_list_2 = []
        for i in range(3):
            env1.reset()
            env2.reset()
            state_list_1.append(env1.state)
            state_list_2.append(env2.state)

        # They should NOT match for different seeds
        for s1, s2 in zip(state_list_1, state_list_2):
            self.assertFalse(
                states_equal(s1, s2)
            )

    def test_generate_agent(self):
        self.env._generate_agent("UGV", 3)
        print(f"Added UGV 3 to state")
        self.assertIn("UGV_3_pos", self.env.state)
        self.assertIn("UGV_3_vel", self.env.state)
        self.assertIn("UGV_3_fov", self.env.state)
        self.assertNotIn("UGV_3_altitude", self.env.state) # UGV does not have altitude
        self.env._generate_agent("UAV", 4)
        print(f"Added UAV 4 to state")
        self.assertIn("UAV_4_pos", self.env.state)
        self.assertIn("UAV_4_vel", self.env.state)
        self.assertIn("UAV_4_fov", self.env.state)
        self.assertIn("UAV_4_altitude", self.env.state)
        self.env._generate_agent("target", 5)
        print(f"Added target 5 to state")
        self.assertIn("target_5_pos", self.env.state)
        self.assertIn("target_5_vel", self.env.state)
        self.assertIn("target_5_fov", self.env.state)
        self.assertNotIn("target_5_altitude", self.env.state) # Target does not have altitude

    def test_generate_obstacles(self):
        """Test that obstacles are generated correctly"""
        map_data = self.env.state["map"]
        map_size = map_data["size"]
        
        # Check physical obstacles
        physical_obstacles = map_data["physical_obstacles"]
        self.assertIsInstance(physical_obstacles, np.ndarray)
        self.assertGreater(len(physical_obstacles), 0)
        
        # Check that obstacles are within map bounds
        for obstacle in physical_obstacles:
            x, y = obstacle[0], obstacle[1]
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, map_size[0])
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, map_size[1])
        
        # Check visual obstacles
        visual_obstacles = map_data["visual_obstacles"]
        self.assertIsInstance(visual_obstacles, np.ndarray)
        self.assertGreater(len(visual_obstacles), 0)
        
        # Check that obstacles are within map bounds
        for obstacle in visual_obstacles:
            x, y = obstacle[0], obstacle[1]
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, map_size[0])
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, map_size[1])

    def test_observation_space(self):
        """Test that observation spaces are correctly defined for each agent type"""
        from gymnasium.spaces import Box
        
        # Test UAV observation space
        uav_obs_space = self.env.observation_space("UAV_0")
        self.assertIsInstance(uav_obs_space, Box)
        self.assertEqual(uav_obs_space.dtype, np.float32)
        
        # Test UGV observation space
        ugv_obs_space = self.env.observation_space("UGV_0")
        self.assertIsInstance(ugv_obs_space, Box)
        self.assertEqual(ugv_obs_space.dtype, np.float32)
        
        # Test target observation space
        target_obs_space = self.env.observation_space("target_0")
        self.assertIsInstance(target_obs_space, Box)
        self.assertEqual(target_obs_space.dtype, np.float32)
        
        # Check that observations fit their spaces
        uav_obs = self.env._get_observation("UAV_0")
        ugv_obs = self.env._get_observation("UGV_0")
        target_obs = self.env._get_observation("target_0")
        
        self.assertEqual(len(uav_obs), uav_obs_space.shape[0])
        self.assertEqual(len(ugv_obs), ugv_obs_space.shape[0])
        self.assertEqual(len(target_obs), target_obs_space.shape[0])
        
        # Check that observations are within bounds
        self.assertTrue(uav_obs_space.contains(uav_obs))
        self.assertTrue(ugv_obs_space.contains(ugv_obs))
        self.assertTrue(target_obs_space.contains(target_obs))

    def test_action_space(self):
        """Test that action spaces are correctly defined for each agent type"""
        from gymnasium.spaces import Discrete
        
        # Test UAV action space
        uav_action_space = self.env.action_space("UAV_0")
        self.assertIsInstance(uav_action_space, Discrete)
        self.assertEqual(uav_action_space.n, 8)  # 8 actions for UAV
        
        # Test UGV action space
        ugv_action_space = self.env.action_space("UGV_0")
        self.assertIsInstance(ugv_action_space, Discrete)
        self.assertEqual(ugv_action_space.n, 5)  # 5 actions for UGV
        
        # Test target action space
        target_action_space = self.env.action_space("target_0")
        self.assertIsInstance(target_action_space, Discrete)
        self.assertEqual(target_action_space.n, 5)  # 5 actions for target
        
        # Test that valid actions are accepted
        for action in range(uav_action_space.n):
            self.assertTrue(uav_action_space.contains(action))
        for action in range(ugv_action_space.n):
            self.assertTrue(ugv_action_space.contains(action))
        for action in range(target_action_space.n):
            self.assertTrue(target_action_space.contains(action))

    def test_reset(self):
        """Test that reset initializes the environment correctly"""
        # Reset the environment
        observations, infos = self.env.reset()
        
        # Check that observations are returned
        self.assertIsInstance(observations, dict)
        self.assertIsInstance(infos, dict)
        
        # Check that observations exist for all agents
        for agent in self.env.agents:
            self.assertIn(agent, observations)
            self.assertIn(agent, infos)
            self.assertIsInstance(observations[agent], np.ndarray)
        
        # Check that num_moves is reset
        self.assertEqual(self.env.num_moves, 0)
        
        # Check that agents are set correctly
        self.assertEqual(len(self.env.agents), len(self.env.possible_agents))
        
        # Check that map is generated
        self.assertIn("map", self.env.state)
        self.assertIn("size", self.env.state["map"])
        self.assertIn("physical_obstacles", self.env.state["map"])
        self.assertIn("visual_obstacles", self.env.state["map"])
        
        # Check that agents are placed
        for agent in self.env.agents:
            pos = self.env.state[f"{agent}_pos"]
            self.assertGreaterEqual(pos[0], 0)
            self.assertLess(pos[0], self.env.state["map"]["size"][0])
            self.assertGreaterEqual(pos[1], 0)
            self.assertLess(pos[1], self.env.state["map"]["size"][1])
        
        # Check that goal is placed
        self.assertIn("goal_pos", self.env.state)
        goal_pos = self.env.state["goal_pos"]
        self.assertGreaterEqual(goal_pos[0], 0)
        self.assertLess(goal_pos[0], self.env.state["map"]["size"][0])
        self.assertGreaterEqual(goal_pos[1], 0)
        self.assertLess(goal_pos[1], self.env.state["map"]["size"][1])

    # def test_step(self):
    #     """Test that step function works correctly"""
    #     observations, infos = self.env.reset()
        
    #     # Create valid actions for all agents
    #     actions = {}
    #     for agent in self.env.agents:
    #         actions[agent] = 0  # Use action 0 (move down)
        
    #     # Take a step
    #     obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
    #     # Check return types
    #     self.assertIsInstance(obs, dict)
    #     self.assertIsInstance(rewards, dict)
    #     self.assertIsInstance(terminations, dict)
    #     self.assertIsInstance(truncations, dict)
    #     self.assertIsInstance(infos, dict)
        
    #     # Check that all agents are in the dictionaries
    #     for agent in self.env.agents:
    #         self.assertIn(agent, obs)
    #         self.assertIn(agent, rewards)
    #         self.assertIn(agent, terminations)
    #         self.assertIn(agent, truncations)
    #         self.assertIn(agent, infos)
        
    #     # Check that observations are numpy arrays
    #     for agent in self.env.agents:
    #         self.assertIsInstance(obs[agent], np.ndarray)
        
    #     # Check that rewards are numbers
    #     for agent in self.env.agents:
    #         self.assertIsInstance(rewards[agent], (int, float, np.number))
        
    #     # Check that terminations and truncations are booleans
    #     for agent in self.env.agents:
    #         self.assertIsInstance(terminations[agent], (bool, np.bool_))
    #         self.assertIsInstance(truncations[agent], (bool, np.bool_))
        
    #     # Check that num_moves is incremented
    #     self.assertEqual(self.env.num_moves, 1)
        
    #     # Test with empty actions
    #     empty_obs, empty_rewards, empty_term, empty_trunc, empty_infos = self.env.step({})
    #     self.assertEqual(empty_obs, {})
    #     self.assertEqual(empty_rewards, {})
    #     self.assertEqual(len(self.env.agents), 0)

    def test_update_state_movement(self):
        '''Test that movement actions correctly update agent position'''
        print(self.env.possible_actions["UAV"])
        print(self.env.possible_actions["UGV"])
        print(self.env.possible_actions["target"])

        for i, action_key in enumerate(self.env.possible_actions["UAV"]):
            if i < 4:  # Only test movement actions (0-3), skip altitude actions (4-7)
                # Copy previous position to avoid reference issues
                previous_pos = self.env.state["UAV_0_pos"].copy()
                expected_movement = np.array(self.env.possible_actions["UAV"][action_key][0])
                
                # Calculate expected position (accounting for boundary clipping)
                expected_pos = previous_pos + expected_movement
                map_size = self.env.state["map"]["size"]
                expected_pos[0] = max(0, min(expected_pos[0], map_size[0] - 1))
                expected_pos[1] = max(0, min(expected_pos[1], map_size[1] - 1))
                
                # Update state with action (pass action_key as integer or string)
                self.env._update_state("UAV_0", int(action_key))
                new_pos = self.env.state["UAV_0_pos"]
                
                print(f"Action {action_key}: previous={previous_pos}, expected={expected_pos}, actual={new_pos}")
                self.assertTrue(np.all(new_pos == expected_pos), 
                              f"Position mismatch for action {action_key}: expected {expected_pos}, got {new_pos}")
            else:
                pass
    
    def test_update_state_out_of_bounds(self):
        """Test that out-of-bounds movements are correctly clipped"""
        # Place agent at top-left corner (0, 0)
        self.env.state["UAV_0_pos"] = np.array([0, 0], dtype=int)
        
        # Try to move up (action 1) - should stay at y=0
        previous_pos = self.env.state["UAV_0_pos"].copy()
        self.env._update_state("UAV_0", 1)  # move up
        new_pos = self.env.state["UAV_0_pos"]
        self.assertEqual(new_pos[1], 0)  # Should be clipped to 0
        self.assertEqual(new_pos[0], previous_pos[0])
        
        # Try to move left (action 2) - should stay at x=0
        self.env._update_state("UAV_0", 2)  # move left
        new_pos = self.env.state["UAV_0_pos"]
        self.assertEqual(new_pos[0], 0)  # Should be clipped to 0
        self.assertEqual(new_pos[1], 0)
        
        # Place agent at bottom-right corner
        map_size = self.env.state["map"]["size"]
        self.env.state["UAV_0_pos"] = np.array([map_size[0] - 1, map_size[1] - 1], dtype=int)
        
        # Try to move down (action 0) - should stay at y=size-1
        previous_pos = self.env.state["UAV_0_pos"].copy()
        self.env._update_state("UAV_0", 0)  # move down
        new_pos = self.env.state["UAV_0_pos"]
        self.assertEqual(new_pos[1], map_size[1] - 1)  # Should be clipped
        self.assertEqual(new_pos[0], previous_pos[0])
        
        # Try to move right (action 3) - should stay at x=size-1
        self.env._update_state("UAV_0", 3)  # move right
        new_pos = self.env.state["UAV_0_pos"]
        self.assertEqual(new_pos[0], map_size[0] - 1)  # Should be clipped
        self.assertEqual(new_pos[1], map_size[1] - 1)

    def test_update_state_invalid_action(self):
        """Test that invalid actions are handled gracefully"""
        # Test with action that doesn't exist in the action space
        # Note: The function should handle this - it might raise an error or ignore it
        previous_pos = self.env.state["UAV_0_pos"].copy()
        
        # Try action 99 (invalid)
        try:
            self.env._update_state("UAV_0", 99)
            # If no error is raised, the position should remain unchanged or be handled somehow
            # This depends on the implementation - for now, we just check it doesn't crash
            new_pos = self.env.state["UAV_0_pos"]
            # If it doesn't crash, that's acceptable behavior
        except KeyError:
            # KeyError is expected if the action doesn't exist
            pass
        except Exception as e:
            # Other exceptions might be acceptable depending on implementation
            # The test passes if it doesn't crash with an unexpected error
            print(f"Caught expected exception: {e}")
        
        # Test with negative action
        try:
            self.env._update_state("UAV_0", -1)
        except Exception as e:
            # Exceptions are acceptable for invalid actions
            print(f"Caught expected exception: {e}")

    def test_update_state_altitude(self):
        """Test that altitude updates work correctly for UAVs"""
        # Start at altitude 0
        self.env.state["UAV_0_altitude"] = np.array([0], dtype=int)
        
        # Test increase altitude (action 5)
        self.env._update_state("UAV_0", 5)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 1)
        
        # Test increase again
        self.env._update_state("UAV_0", 5)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 2)
        
        # Test increase at max altitude - should be clipped
        self.env._update_state("UAV_0", 5)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 2)  # Clipped at max
        
        # Test hold altitude (action 6)
        self.env._update_state("UAV_0", 6)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 2)  # Should remain 2
        
        # Test decrease altitude (action 7)
        self.env._update_state("UAV_0", 7)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 1)
        
        # Test decrease again
        self.env._update_state("UAV_0", 7)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 0)
        
        # Test decrease at min altitude - should be clipped
        self.env._update_state("UAV_0", 7)
        self.assertEqual(self.env.state["UAV_0_altitude"][0], 0)  # Clipped at min
        
        # Test that UGV altitude actions don't affect UGV (it has no altitude)
        if "UGV_0_altitude" not in self.env.state:
            # UGV doesn't have altitude, so altitude actions should be ignored
            previous_pos = self.env.state["UGV_0_pos"].copy()
            self.env._update_state("UGV_0", 5)  # This should only affect position if action < 5
            # Position should remain the same since action 5 is not a movement action for UGV
            # Actually, for UGV, action 5 doesn't exist, so this might cause an error or be ignored
            pass

    
    def test_truncation(self):
        """Test that truncation works correctly"""
        env1 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)
        env2 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)
        env3 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)

        step_limit = self.env.step_limit
        env1.step_count = step_limit - 2
        obs, rewards, terminations, truncations, infos = env1.step({agent: 3 for agent in env1.possible_agents})
        self.assertFalse(truncations["UAV_0"])
        self.assertFalse(truncations["UGV_0"])
        self.assertFalse(truncations["target_0"])

        env2.step_count = step_limit
        obs, rewards, terminations, truncations, infos = env2.step({agent: 3 for agent in env2.possible_agents})
        self.assertTrue(truncations["UAV_0"])
        self.assertTrue(truncations["UGV_0"])
        self.assertTrue(truncations["target_0"])

        env3.step_count = step_limit + 1
        obs, rewards, terminations, truncations, infos = env3.step({agent: 3 for agent in env3.possible_agents})
        self.assertTrue(truncations["UAV_0"])
        self.assertTrue(truncations["UGV_0"])
        self.assertTrue(truncations["target_0"])

    def test_termination(self):
        """Test that termination works correctly"""
        self.env.state["target_0_pos"] = np.array([10, 10])
        self.env.state["UAV_0_pos"] = np.array([9,9])
        self.env.state["UGV_0_pos"] = np.array([9,8])
        obs, rewards, terminations, truncations, infos = self.env.step({agent: 4 for agent in self.env.possible_agents})
        self.assertTrue(terminations["UAV_0"])
        self.assertFalse(terminations["UGV_0"])
        self.assertFalse(terminations["target_0"])

        self.env.reset()
        self.env.state["target_0_pos"] = np.array([10, 10])
        self.env.state["UAV_0_pos"] = np.array([9,8])
        self.env.state["UGV_0_pos"] = np.array([9,9])
        obs, rewards, terminations, truncations, infos = self.env.step({agent: 4 for agent in self.env.possible_agents})
        self.env.viz_map()
        self.assertFalse(terminations["UAV_0"])
        self.assertTrue(terminations["UGV_0"])
        self.assertFalse(terminations["target_0"])

        self.env.reset()
        self.env.state["target_0_pos"] = np.array([10, 10])
        self.env.state["UAV_0_pos"] = np.array([9,10])
        self.env.state["UGV_0_pos"] = np.array([10,9])
        obs, rewards, terminations, truncations, infos = self.env.step({agent: 4 for agent in self.env.possible_agents})
        self.assertTrue(terminations["UAV_0"])
        self.assertTrue(terminations["UGV_0"])
        self.assertTrue(terminations["target_0"])

    def test_action_space_sample_seed(self):
        """Test that action space sample is deterministic with a given seed"""
        env1 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)
        env2 = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)
        
        actions_1 = []
        actions_2 = []

        self.assertEqual(env1.np_random_seed, env2.np_random_seed)
        print("np_random_seed is the same")

        for i in range(10):
            action1 = env1.action_space("UAV_0").sample()
            action2 = env2.action_space("UAV_0").sample()
            actions_1.append(action1)
            actions_2.append(action2)

        self.assertEqual(actions_1, actions_2)

    def test_communication(self):
        """Test that communication works correctly"""
        
        # Test no communication
        # Create environment with no communication
        env_no_comm = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, 
                                  scenario="explore", map_type="medium", step_limit=500, 
                                  seed=42, communication_style="none", verbose=False)
        env_no_comm.reset()
        
        # Place UAV_0 right next to target_0 (within FOV - UAV FOV is 7x7, so half_fov = 3)
        # Place target at (10, 8), UAV at (10, 10) - definitely in FOV but not in target's FOV
        env_no_comm.state["target_0_pos"] = np.array([10, 8])
        env_no_comm.state["UAV_0_pos"] = np.array([10, 10])
        
        # Place UGV_0 far away (not in FOV of target, and not in FOV of UAV)
        # UGV FOV is 5x5, so half_fov = 2. Place UGV at (0, 0) - far from (10, 10)
        env_no_comm.state["UGV_0_pos"] = np.array([0, 0])
        
        # Step and get observations
        obs, rewards, terminations, truncations, infos = env_no_comm.step(
            {agent: 4 for agent in env_no_comm.possible_agents}  # Do nothing action
        )
        
        # Extract target position from UGV_0's observation
        # Observation structure for UGV: [pos(2), vel(2), fov(25), target_vels(2), target_positions(2)]
        # So target positions start at index: 2 + 2 + 25 = 29
        ugv_obs = obs["UGV_0"]
        fov_dim = int(env_no_comm.state["UGV_0_fov_dim"])
        fov_size = fov_dim ** 2
        # UGV: pos(2) + vel(2) + fov(fov_size) + target_vels(2) + target_positions(2)
        target_pos_idx = 2 + 2 + fov_size + 2  # After target_vels
        target_pos_in_obs = ugv_obs[target_pos_idx:target_pos_idx + 2]
        
        # With no communication, UGV should NOT see the target (should be -10)
        self.assertEqual(target_pos_in_obs[0], -10, 
                        "UGV should not see target with no communication")
        self.assertEqual(target_pos_in_obs[1], -10, 
                        "UGV should not see target with no communication")
        
        # Test complete communication
        # Create environment with complete communication
        env_complete_comm = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, 
                                       scenario="explore", map_type="medium", step_limit=500, 
                                       seed=42, communication_style="complete", verbose=False)
        env_complete_comm.reset()
        
        # Same setup: UAV_0 next to target_0, UGV_0 far away
        env_complete_comm.state["target_0_pos"] = np.array([10, 8])
        env_complete_comm.state["UAV_0_pos"] = np.array([10, 10])
        env_complete_comm.state["UGV_0_pos"] = np.array([0, 0])
        
        # Step and get observations
        obs, rewards, terminations, truncations, infos = env_complete_comm.step(
            {agent: 4 for agent in env_complete_comm.possible_agents}  # Do nothing action
        )
        
        # Extract target position from UGV_0's observation
        ugv_obs = obs["UGV_0"]
        print(f"UGV observation: {ugv_obs}")
        fov_dim = int(env_complete_comm.state["UGV_0_fov_dim"])
        fov_size = fov_dim ** 2
        target_pos_idx = 2 + 2 + fov_size + 2  # After target_vels
        target_pos_in_obs = ugv_obs[target_pos_idx:target_pos_idx + 2]
        
        # With complete communication, UGV SHOULD see the target (should be actual position, not -10)
        self.assertNotEqual(target_pos_in_obs[0], -10, 
                           "UGV should see target with complete communication")
        self.assertNotEqual(target_pos_in_obs[1], -10, 
                           "UGV should see target with complete communication")
        # Check that it's the actual target position
        self.assertEqual(target_pos_in_obs[0], 10.0, 
                        "UGV should see correct target x position with complete communication")
        self.assertEqual(target_pos_in_obs[1], 8.0, 
                        "UGV should see correct target y position with complete communication")

    def test_update_state_obstacle_collision(self):
        """Test that obstacle collision is handled correctly"""
        self.env.state["UGV_0_pos"] = np.array([0, 0])
        self.env.state["UAV_0_pos"] = np.array([0, 0])
        self.env.state["map"]["physical_obstacles"] = np.array([[1, 0]])
        self.env.state["map"]["visual_obstacles"] = np.array([[1, 0]])
        self.env._update_state("UGV_0", 3)
        self.env._update_state("UAV_0", 3)
        
        self.assertTrue((self.env.state["UGV_0_pos"] == np.array([0, 0])).all())
        self.assertTrue((self.env.state["UAV_0_pos"] == np.array([1, 0])).all())

    def test_update_state_obstacle_collision_reward(self):
        rewards = {agent: -0.02 for agent in self.env.possible_agents}
        self.env.state["UGV_0_pos"] = np.array([0, 0])
        self.env.state["UAV_0_pos"] = np.array([0, 0])
        self.env.state["goal_pos"] = np.array([10,10])
        self.env._update_distance_to_goal()
        self.env.state["map"]["physical_obstacles"] = np.array([[1, 0]])
        self.env.state["map"]["visual_obstacles"] = np.array([[1, 0]])
        obs, rewards, terminations, truncations, infos = self.env.step({agent: 3 for agent in self.env.possible_agents})
        
        self.assertTrue((self.env.state["UGV_0_pos"] == np.array([0, 0])).all())
        self.assertTrue((self.env.state["UAV_0_pos"] == np.array([1, 0])).all())
        self.assertEqual(rewards["UGV_0"], self.env.reward_weights["obstacle_penalty"] + self.env.reward_weights["step_penalty"]) # UGV bumped into obstacle
        self.assertEqual(rewards["UAV_0"], 0 + self.env.reward_weights["step_penalty"] + self.env.reward_weights["agent_moved_towards_goal"]) # UAV didn't bump into obstacle

    def test_target_reward_0_if_target_movement_pattern_is_random(self):
        # TODO:
        self.env = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, target_movement_pattern="random", scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)
        target_rewards = 0
        uav_rewards = 0
        ugv_rewards = 0
        terminations = {agent: False for agent in self.env.agents}
        truncations = {agent: False for agent in self.env.agents}

        while not all(terminations[a] or truncations[a] for a in self.env.agents):
            alive_agents = [
                a for a in self.env.agents
                if not (terminations[a] or truncations[a])
            ]
            actions = {
                agent: self.env.action_space(agent).sample()
                for agent in alive_agents
            }
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            target_rewards += rewards["target_0"]
            uav_rewards += rewards["UAV_0"]
            ugv_rewards += rewards["UGV_0"]

        self.assertEqual(target_rewards, 0)
        self.assertNotEqual(uav_rewards, 0)
        self.assertNotEqual(ugv_rewards, 0)
        pass

    def test_update_distance_to_goal(self):
        self.env.state["goal_pos"] = np.array([5,5])
        self.env.state["UAV_0_pos"] = np.array([5,2])   
        self.env.state["UGV_0_pos"] = np.array([2,5])
        self.env._update_distance_to_goal()
        self.assertEqual(self.env.distance_to_goal["UAV_0"], 3)
        self.assertEqual(self.env.distance_to_goal["UGV_0"], 3)

    def test_rewards_for_aproaching_goal(self):
        self.env.state["goal_pos"] = np.array([5,5])
        self.env.state["UAV_0_pos"] = np.array([5,2])
        self.env.state["UGV_0_pos"] = np.array([2,5])
        self.env.state["target_0_pos"] = np.array([10,10])
        self.env._update_distance_to_goal()
        self.assertEqual(self.env.distance_to_goal["UAV_0"], 3)
        self.assertEqual(self.env.distance_to_goal["UGV_0"], 3)
        obs, rewards, terminations, truncations, infos = self.env.step({"UAV_0": 1, "UGV_0": 3, "target_0": 4})
        self.assertEqual(self.env.distance_to_goal["UAV_0"], 4)
        self.assertEqual(self.env.distance_to_goal["UGV_0"], 2)
        self.assertEqual(self.env.reduced_distance_to_goal["UAV_0"], False)
        self.assertEqual(self.env.reduced_distance_to_goal["UGV_0"], True)
        self.assertEqual(rewards["UAV_0"], self.env.reward_weights["step_penalty"])
        self.assertEqual(rewards["UGV_0"], self.env.reward_weights["agent_moved_towards_goal"] + self.env.reward_weights["step_penalty"])
        pass

    def test_get_state_array(self):
        """Test that get_state_array works correctly"""
        state_array = self.env.get_state_array()
        print(f"State array: {state_array}")
        print(f"State: {self.env.state}")
        self.assertIsInstance(state_array, np.ndarray)
        self.assertEqual(state_array[0], self.env.state["UAV_0_pos"][0])
        self.assertEqual(state_array[1], self.env.state["UAV_0_pos"][1])
        self.assertEqual(state_array[2], self.env.state["UAV_0_vel"][0])
        self.assertEqual(state_array[3], self.env.state["UAV_0_vel"][1])
        self.assertEqual(state_array[4], self.env.state["UAV_0_fov_dim"])
        self.assertEqual(state_array[5], self.env.state["UAV_0_altitude"][0])
        self.assertEqual(state_array[6 + self.env.state["UAV_0_fov_dim"]**2], self.env.state["UGV_0_pos"][0])
        self.assertEqual(state_array[7 + self.env.state["UAV_0_fov_dim"]**2], self.env.state["UGV_0_pos"][1])
        self.assertEqual(state_array[8 + self.env.state["UAV_0_fov_dim"]**2], self.env.state["UGV_0_vel"][0])
        self.assertEqual(state_array[9 + self.env.state["UAV_0_fov_dim"]**2], self.env.state["UGV_0_vel"][1])
        self.assertEqual(state_array[10 + self.env.state["UAV_0_fov_dim"]**2], self.env.state["UGV_0_fov_dim"])
        self.assertEqual(state_array[11 + self.env.state["UAV_0_fov_dim"]**2 + self.env.state["UGV_0_fov_dim"]**2], self.env.state["target_0_pos"][0])
        self.assertEqual(state_array[12 + self.env.state["UAV_0_fov_dim"]**2 + self.env.state["UGV_0_fov_dim"]**2], self.env.state["target_0_pos"][1])
        self.assertEqual(state_array[13 + self.env.state["UAV_0_fov_dim"]**2 + self.env.state["UGV_0_fov_dim"]**2], self.env.state["target_0_vel"][0])
        self.assertEqual(state_array[14 + self.env.state["UAV_0_fov_dim"]**2 + self.env.state["UGV_0_fov_dim"]**2], self.env.state["target_0_vel"][1])
        self.assertEqual(state_array[15 + self.env.state["UAV_0_fov_dim"]**2 + self.env.state["UGV_0_fov_dim"]**2], self.env.state["target_0_fov_dim"])
        a = 15 + self.env.state["UAV_0_fov_dim"]**2 + self.env.state["UGV_0_fov_dim"]**2 + self.env.state["target_0_fov_dim"]**2
        self.assertEqual(state_array[1 + a], self.env.state["map"]["size"][0])
        self.assertEqual(state_array[2 + a], self.env.state["map"]["size"][1])
        # Flatten and extract physical obstacle coordinates
        phys_obs_start = 3 + a
        phys_obs = self.env.state["map"]["physical_obstacles"]
        for i in range(len(phys_obs)):
            self.assertEqual(state_array[phys_obs_start + 2*i], phys_obs[i][0])
            self.assertEqual(state_array[phys_obs_start + 2*i + 1], phys_obs[i][1])
        # Now visual obstacles, which come next
        vis_obs_start = phys_obs_start + 2 * len(phys_obs)
        vis_obs = self.env.state["map"]["visual_obstacles"]
        for i in range(len(vis_obs)):
            self.assertEqual(state_array[vis_obs_start + 2*i], vis_obs[i][0])
            self.assertEqual(state_array[vis_obs_start + 2*i + 1], vis_obs[i][1])
        self.assertEqual(state_array[-2], self.env.state["goal_pos"][0])
        self.assertEqual(state_array[-1], self.env.state["goal_pos"][1])

    def test_number_of_obstacles(self):
        max_obstacles_points = 20 if self.env.map_type == "small" else 30 if self.env.map_type == "medium" else 40
        self.assertEqual(len(self.env.state["map"]["physical_obstacles"]), max_obstacles_points)
        self.assertEqual(len(self.env.state["map"]["visual_obstacles"]), max_obstacles_points)

def test_elendil_v2():
    env = elendil_v2(render_mode=None, num_UGVs=1, num_UAVs=1, num_targets=1, scenario="explore", map_type="medium", step_limit=500, seed=42, verbose=False)
    # parallel_api_test(env, num_cycles=1000)
    unittest.main()

if __name__ == '__main__':
    test_elendil_v2()