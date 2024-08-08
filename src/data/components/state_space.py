from typing import List, Tuple
from collections import deque


import random
import torch
import numpy as np

# Set a fixed seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class StateSpace:
    def __init__(self, seed: int, num_fcs: int, lt_values: List[int], rp_arrays: List[List[int]], forecast_horizon: int):
        # seed setting in state space
        self.rng = np.random.default_rng(seed)  # Create a separate RNG

        self.num_fcs = num_fcs
        self.lt_values = torch.tensor(lt_values, dtype=torch.int32)
        self.rp_arrays = torch.tensor(rp_arrays, dtype=torch.int32)
        self.forecast_horizon = forecast_horizon

        assert self.rp_arrays.shape == (num_fcs, forecast_horizon), "RP arrays shape mismatch"

        self.current_timestep = 0

        # Padding for past sales to get rs_lt during S0
        self.max_lt = max(lt_values)
        self.sales_history = torch.zeros((self.num_fcs, forecast_horizon + self.max_lt), dtype=torch.float32)
        self.action_history = torch.zeros((self.num_fcs, forecast_horizon), dtype=torch.float32)
        # Making sure to replicate the situation where there were prior actions before S0 to get action_pipeline


        # Random initial sales when env is reset for prior timesteps
        # self.on_hand = torch.ceil(torch.rand(self.num_fcs, dtype=torch.float32) * 100)
        self.sales_history[:, -self.max_lt:] = torch.randint(0, 20, (self.num_fcs, self.max_lt)).float()


        self.zero_state()
        self.endpoint = self.forecast_horizon - self.max_lt
        self.reset_count = 0

        self.forecast = None  # To be set by the environment

    def set_seed(self, seed: int):
        """Set a new seed for the random number generator"""
        self.rng = np.random.default_rng(seed)

    def reinitialize(self):
        """Reinitialize the state with the current RNG"""
        self.zero_state()

    def zero_state(self):
        """
        Create the initial state to fill up with actions pipeline to reset after episode is done
        :return:
        """
        self.current_timestep = 0
        # self.on_hand = torch.ceil(torch.rand(self.num_fcs, dtype=torch.float32) * 100)

        self.on_hand = torch.from_numpy(self.rng.uniform(0, 100, size=self.num_fcs)).float().ceil()
        self.action_pipeline = [deque(maxlen=forecast_horizon) for _ in range(self.num_fcs)]

        # Random initial on-hand inventory
        for fc in range(self.num_fcs):
            for _ in range(2):  # Initialize with 2 prior random actions
                action = self.rng.integers(0, 10)
                # Any prior RP. These will get replenished based on LT
                timestamp = self.current_timestep - self.rng.integers(1, self.lt_values[fc])
                self.action_pipeline[fc].append((action, timestamp))


    def set_forecast(self, forecast: torch.Tensor):
        """Set the forecast for all FCs."""
        assert forecast.shape == (self.num_fcs, self.forecast_horizon)
        self.forecast = forecast

    def update(self, sales: torch.Tensor, actions: torch.Tensor):
        """Update the state with new sales and actions."""
        assert sales.shape == actions.shape == (self.num_fcs,)

        # Update histories
        self.sales_history = torch.roll(self.sales_history, shifts=-1, dims=1)
        self.sales_history[:, -1] = sales

        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.action_history[:, -1] = actions


        # Update on-hand inventory and action pipeline
        for fc in range(self.num_fcs):
            lt = self.lt_values[fc]

            # Process replenishments
            repl_received = 0
            # Previous timestep repl_received to update the oh_curr
            while self.action_pipeline[fc] and self.action_pipeline[fc][0][1] + lt <= self.current_timestep:
                action, _ = self.action_pipeline[fc].popleft()
                repl_received += action

            # Update on-hand inventory
            self.on_hand[fc] += repl_received
            self.on_hand[fc] -= sales[fc]

            # Add new action to pipeline
            self.action_pipeline[fc].append((actions[fc].item(), self.current_timestep))

        self.current_timestep += 1

    def get_state(self) -> torch.Tensor:
        """Compute and return the current state for all FCs."""
        states = []

        for fc in range(self.num_fcs):
            lt = self.lt_values[fc]
            rp = self.rp_arrays[fc]

            oh_curr = self.on_hand[fc]  # sales have already been deducted. Updated for new timestep

            # Replenishment should be received only if the LT for action in pipeline matches to today
            # Calculate replenishment received (if any) for this timestep
            repl_received = sum(action for action, timestamp in self.action_pipeline[fc] if timestamp + lt == self.current_timestep)

            # On-hands after replenishment
            oh_repl = oh_curr + repl_received

            # Agent needs to see how much is left in the action pipeline yet to replen before taking an action.
            # Done after current timestamp replen is removed
            action_pipeline = sum(action for action, _ in self.action_pipeline[fc])


            dem_lt = self.forecast[fc, self.current_timestep:self.current_timestep + lt].sum()

            # Calculate demand during coverage period (CP) using the RP array
            cp_start = self.current_timestep + lt
            # cp_end = min(cp_start + torch.where(rp[cp_start:] == 1)[0][0].item() + 1, self.forecast_horizon)
            next_rp = torch.where(rp[cp_start:] == 1)[0]
            if len(next_rp) > 0:
                cp_end = min(cp_start + next_rp[0].item() + 1, self.forecast_horizon)
            else:
                cp_end = self.forecast_horizon

            dem_cp = self.forecast[fc, cp_start:cp_end].sum()

            # Realized sales for past lt time-steps
            rs_lt = self.sales_history[fc, -lt:].sum()

            # Calculate days left till next decision
            next_decision = torch.where(rp[self.current_timestep:] == 1)[0]
            days_left = next_decision[0].item() if len(next_decision) > 0 else 0

            # print(f"FC {fc}: oh_curr = {oh_curr}, oh_repl = {oh_repl}, repl_received = {repl_received}, action_pipeline = {action_pipeline}")

            fc_state = torch.tensor([
                oh_curr, action_pipeline, repl_received, oh_repl, dem_cp, dem_lt, rs_lt , days_left
            ], dtype=torch.float32)

            states.append(fc_state)

        return torch.cat(states) #states #torch.cat(states)

    def get_state_dim(self) -> int:
        """Return the dimension of the state vector."""
        return 8 * self.num_fcs

    def get_current_time_step(self):
        """
        Returns the time step of the state
        :return:
        """
        return self.current_timestep


def reset(seed,inventory_state):
    if reset_count>0:
        inventory_state.set_seed(seed)
        inventory_state.reinitialize()

    forecast = torch.ceil(torch.rand((num_fcs, forecast_horizon)) * 10)  # Random forecast for demonstration
    # print(f"Forecast \n: {forecast}")
    # print(f"Mapped Demand \n: {mapped_demand}")
    inventory_state.set_forecast(forecast)

    return inventory_state.get_state()

def step(state,demand):

    inv_begin = state[0].reshape(-1, 1).item()
    inv_repl = state[3].reshape(-1, 1).item()
    repl_received = inv_repl - inv_begin
    # print("\n",inventory_state.action_pipeline)
    # print(f"inv_begin: {inv_begin}, inv_repl :{inv_repl}, repl:{repl_received}")

    sales = np.minimum(inv_repl, demand)  # Random sales
    actions = torch.ceil(torch.rand(num_fcs) * 20)  # Random actions
    inventory_state.update(torch.Tensor([sales]), actions)

    next_state = inventory_state.get_state()
    print(f"Next state: {next_state}, sales:{ sales}, action :{actions.item()}, repl:{repl_received}")
    return next_state

if __name__=="__main__":
    # Initialize the StateSpace
    num_fcs = 1
    lt_values = [3]
    forecast_horizon = 30
    reset_count = 0
    # Create RP arrays (example: review every 3 days for all FCs)
    rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(forecast_horizon)] for _ in range(num_fcs)]
    # Set the forecast (normally provided by the environment)
    mapped_demand = torch.ceil(torch.rand((num_fcs, forecast_horizon)) * 9).numpy()

    seed = 0
    # Initialize
    inventory_state = StateSpace(seed, num_fcs, lt_values, rp_arrays, forecast_horizon)
    #state = inventory_state.get_state()


    # Simulate a few timesteps
    for i in range(100):
        state = reset(seed,inventory_state)
        print(f"Current state: {state}")
        for _ in range(30):
            next_state = step(state,mapped_demand[0][_].item())
            state= next_state
        print("\n\n\n")
        reset_count += 1
        seed = seed + reset_count


    # # Initialize the StateSpace
    #     num_fcs = 1
    #     lt_values = [2]
    #     forecast_horizon = 30
    #
    #     # Create RP arrays (example: review every 3 days for all FCs)
    #     rp_arrays = [[1 if (i + 1) % 3 == 0 else 0 for i in range(forecast_horizon)] for _ in range(num_fcs)]
    #
    #     seed = 43
    #     state_space = StateSpace(seed, num_fcs, lt_values, rp_arrays, forecast_horizon)
    #
    #     # Set the forecast (normally provided by the environment)
    #     forecast = torch.ceil(torch.rand((num_fcs, forecast_horizon))) * 10  # Random forecast for demonstration
    #
    #
    #     state_space.set_seed(seed)
    #     state_space.set_forecast(forecast)
    #
    #     print(state_space.action_pipeline)
    #
    #     # Simulate a few timesteps
    #     for _ in range(30):
    #         sales = torch.ceil(torch.rand(num_fcs) * 10)  # Random sales
    #         actions = torch.ceil(torch.rand(num_fcs) * 20)  # Random actions
    #         state_space.update(sales, actions)
    #
    #         # print("sales for 3 FCs:" , sales)
    #         # print("actions", actions)
    #         current_state = state_space.get_state()
    #         print(f"Current state: {current_state}")
    #         # print(f"State dimension: {state_space.get_state_dim()}")
    #
    #     seed +=3
    #     print(seed)
    #     # state_space.set_seed(seed)
    #     # Update the seed and reinitialize the existing state space
    #     state_space.set_seed(seed)
    #     state_space.reinitialize()
    #     print("\n\n Using new seed \n\n")
    #
    #
    #     # Simulate a few timesteps
    #     for _ in range(30):
    #         sales = torch.ceil(torch.rand(num_fcs) * 10)  # Random sales
    #         actions = torch.ceil(torch.rand(num_fcs) * 20)  # Random actions
    #         state_space.update(sales, actions)
    #
    #         # print("sales for 3 FCs:" , sales)
    #         # print("actions", actions)
    #         current_state = state_space.get_state()
    #         print(f"Current state: {current_state}")
    #         # print(f"State dimension: {state_space.get_state_dim()}")