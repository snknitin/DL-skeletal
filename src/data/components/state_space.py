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
    def __init__(self, num_fcs: int, lt_values: List[int], rp_arrays: List[List[int]], forecast_horizon: int):
        self.num_fcs = num_fcs
        self.lt_values = torch.tensor(lt_values, dtype=torch.int32)
        self.rp_arrays = torch.tensor(rp_arrays, dtype=torch.int32)
        self.forecast_horizon = forecast_horizon

        assert self.rp_arrays.shape == (num_fcs, forecast_horizon), "RP arrays shape mismatch"

        self.current_timestep = 0

        # Padding for past sales to get rs_lt during S0
        self.max_lt = max(lt_values)
        self.sales_history = torch.zeros((num_fcs, forecast_horizon + self.max_lt), dtype=torch.float32)
        # Random initial sales when env is reset for prior timesteps
        self.sales_history[:, -self.max_lt:] = torch.randint(0, 20, (num_fcs, self.max_lt)).float()

        # Random initial on-hand inventory
        self.on_hand = torch.ceil(torch.rand(num_fcs, dtype=torch.float32) * 100)
        # self.on_hand = torch.randint(50, 200, (num_fcs,)).float()

        self.action_history = torch.zeros((num_fcs, forecast_horizon), dtype=torch.float32)
        # self.action_pipeline = torch.zeros((num_fcs, forecast_horizon), dtype=torch.float32)

        # Making sure to replicate the situation where there were prior actions before S0 to get action_pipeline
        self.action_pipeline = [deque(maxlen=forecast_horizon) for _ in range(num_fcs)]
        for fc in range(num_fcs):
            for _ in range(2):  # Initialize with 2 prior random actions
                action = torch.randint(0, 2, (1,)).item()
                # Any prior RP. These will get replenished based on LT
                timestamp = self.current_timestep - torch.randint(1, lt_values[fc], (1,)).item()
                self.action_pipeline[fc].append((action, timestamp))

        self.forecast = None  # To be set by the environment

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


if __name__=="__main__":
    # Initialize the StateSpace
    num_fcs = 3
    lt_values = [2, 2, 2]
    forecast_horizon = 30

    # Create RP arrays (example: review every 3 days for all FCs)
    rp_arrays = [[1 if (i + 1) % 3 == 0 else 0 for i in range(forecast_horizon)] for _ in range(num_fcs)]

    state_space = StateSpace(num_fcs, lt_values, rp_arrays, forecast_horizon)

    # Set the forecast (normally provided by the environment)
    forecast = torch.ceil(torch.rand((num_fcs, forecast_horizon))) * 10  # Random forecast for demonstration
    state_space.set_forecast(forecast)

    # Simulate a few timesteps
    for _ in range(10):
        sales = torch.ceil(torch.rand(num_fcs) * 10)  # Random sales
        actions = torch.ceil(torch.rand(num_fcs) * 20)  # Random actions
        state_space.update(sales, actions)

        # print("sales for 3 FCs:" , sales)
        # print("actions", actions)
        current_state = state_space.get_state()
        print(f"Current state: {current_state}")
        print(f"State dimension: {state_space.get_state_dim()}")