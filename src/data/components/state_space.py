import pickle
from typing import List, Tuple, Dict
from collections import deque


import random
import torch
import numpy as np
import bisect
import rootutils
from torch import Tensor

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
    def __init__(self, seed: int, num_fcs: int, lt_ecdfs: List[List[Tuple[int, float]]], rp_arrays: List[List[int]], forecast_horizon: int):
        # seed setting in state space
        self.rng = torch.Generator().manual_seed(seed)


        self.num_fcs = num_fcs
        self.lt_ecdfs = [self.prepare_ecdf(ecdf) for ecdf in lt_ecdfs]
        self.expected_lt = [self.calculate_expected_lt(ecdf) for ecdf in self.lt_ecdfs]
        self.pipeline_start_idx = max(self.expected_lt)
        self.rp_arrays = torch.tensor(rp_arrays, dtype=torch.int32)
        self.forecast_horizon = forecast_horizon

        assert self.rp_arrays.shape == (num_fcs, forecast_horizon), "RP arrays shape mismatch"

        self.current_timestep = 0

        # Padding for past sales to get rs_lt during S0
        self.max_lt = int(max(param[0] for param in lt_ecdfs[0]))  # Approximate max LT
        self.lbw = 2  # Look back window for x timesteps -> Make corresponding changes in get_state_dim and obs_size in Multi_FC_OT
        self.max_pending_actions = 5
        self.sales_history = torch.zeros((self.num_fcs, forecast_horizon + self.max_lt), dtype=torch.float32)
        self.action_history = torch.zeros((self.num_fcs, forecast_horizon + self.max_lt), dtype=torch.float32)
        self.multiplier_history = torch.zeros((self.num_fcs, forecast_horizon + self.max_lt), dtype=torch.float32)

        # Making sure to replicate the situation where there were prior actions before S0 to get action_pipeline


        # Random initial sales when env is reset for prior timesteps
        # self.on_hand = torch.ceil(torch.rand(self.num_fcs, dtype=torch.float32) * 100)
        self.sales_history[:, -self.max_lt:] = torch.randint(0, 20, (self.num_fcs, self.max_lt))


        self.zero_state()
        self.endpoint = self.forecast_horizon - self.max_lt
        self.reset_count = 0

        self.forecast = None  # To be set by the environment

    def set_seed(self, seed: int):
        """Set a new seed for the random number generator"""
        # self.rng = np.random.default_rng(seed)
        self.rng = torch.Generator().manual_seed(seed)

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
        self.on_hand = torch.randint(0, 100, (self.num_fcs,)).float()

        # Vectorized action pipeline
        self.action_pipeline = torch.zeros((self.num_fcs, self.forecast_horizon+ self.pipeline_start_idx, 4))
        # To figure out the in-transit vs executed
        self.pipeline_mask = torch.zeros((self.num_fcs, self.forecast_horizon+ self.pipeline_start_idx), dtype=torch.bool)

        # Random initial on-hand inventory
        for fc in range(self.num_fcs):
            for t in range(self.expected_lt[fc]):  # Initialize with 2 prior random actions
            # for t in range(self.lbw):
                action = torch.randint(0, 3, (1,)).item()
                self.action_history[:, -1] = action
                timestamp = self.current_timestep - (self.expected_lt[fc]-t) # logic needs to be updated for review period logic
                lt = self.sample_lead_time(fc)

                while timestamp + lt < 0:  # Ensure prior actions are executed after t=0
                        lt = self.sample_lead_time(fc)
                # lt = self.sample_constrained_lead_time(fc, timestamp)
                # print("Initial pipeline")
                # print(fc,lt)
                multiplier = int(self.sales_history[fc, -self.lbw:].mean().item()) # Adding this line

                # Use t as the index directly, no offset needed here
                idx = t
                self.action_pipeline[fc, idx] = torch.tensor([action, timestamp, lt, multiplier])
                self.pipeline_mask[fc, idx] = True

                self.action_history[fc, -1] = action
                self.multiplier_history[fc, -1] = multiplier

                # Any prior RP. These will get replenished based on LT
                # self.action_pipeline[fc].append((action, timestamp,lt,multiplier))

    def prepare_ecdf(self, ecdf: List[Tuple[int, float]]) -> dict[str, Tensor]:
        lt_values, probabilities = zip(*sorted(ecdf))
        return {
            'lt_values': torch.tensor(lt_values, dtype=torch.float32),
            'cum_probabilities': torch.tensor(probabilities, dtype=torch.float32)
        }

    def sample_lead_time(self, fc: int) -> int:
        ecdf = self.lt_ecdfs[fc]
        r = torch.rand(1,generator=self.rng).item()
        idx = torch.searchsorted(ecdf['cum_probabilities'], r).item()
        return ecdf['lt_values'][idx].item()

    def calculate_expected_lt(self, ecdf: Dict[str, torch.Tensor]) -> int:
        lt_values = ecdf['lt_values']
        # probabilities = [ecdf['cum_probabilities'][0]] + [
        #     ecdf['cum_probabilities'][i] - ecdf['cum_probabilities'][i - 1] for i in
        #     range(1, len(ecdf['cum_probabilities']))]
        probabilities = torch.diff(ecdf['cum_probabilities'], prepend=torch.tensor([0.0]))
        expected_lt = torch.sum(lt_values * probabilities)
        # expected_lt = sum(lt * prob for lt, prob in zip(lt_values, probabilities))
        return round(expected_lt.item())

    def sample_constrained_lead_time(self, fc: int, current_timestamp: int) -> int:
        # if self.current_timestep==0:
        if not self.action_pipeline[fc]:
            lt = self.sample_lead_time(fc)
            while current_timestamp+lt< self.current_timestep:
                 lt = self.sample_lead_time(fc)
            return lt

        # return self.sample_lead_time(fc)

        last_action = self.action_pipeline[fc][-1]
        last_replenishment_time = last_action[1] + last_action[2]
        min_lead_time = max(1, last_replenishment_time - current_timestamp)

        attempts = 0
        max_attempts = 100  # Prevent infinite loop
        while attempts < max_attempts:
            lt = self.sample_lead_time(fc)
            if lt >= min_lead_time and current_timestamp + lt >= last_replenishment_time:
                return lt
            attempts += 1

        # If we couldn't find a suitable lead time, return the minimum acceptable one
        return min_lead_time

    def set_forecast(self, forecast: torch.Tensor):
        """Set the forecast for all FCs."""
        assert forecast.shape == (self.num_fcs, self.forecast_horizon)
        self.forecast = forecast

    @torch.no_grad()
    def update(self, sales: torch.Tensor, actions: torch.Tensor, multiplier:torch.Tensor, repl_received: torch.Tensor):
        """Update the state with new sales and actions."""
        assert sales.shape == actions.shape == (self.num_fcs,)


        # Update histories
        self.sales_history = torch.roll(self.sales_history, shifts=-1, dims=1)
        self.sales_history[:, -1] = sales

        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.action_history[:, -1] = actions

        self.multiplier_history = torch.roll(self.multiplier_history, shifts=-1, dims=1)
        self.multiplier_history[:, -1] = multiplier


        # # Process replenishments and update inventory - [action, timestamp, lt, multiplier]
        execution_mask = (self.action_pipeline[:, :, 1] + self.action_pipeline[:, :,2] == self.current_timestep) & self.pipeline_mask


        # Update on-hand inventory
        # After updating on-hand inventory
        new_on_hand = self.on_hand + repl_received - sales
        assert torch.all(new_on_hand >= 0), f"Negative on-hand inventory detected: {new_on_hand}"
        self.on_hand = torch.max(new_on_hand, torch.zeros_like(new_on_hand))  # Ensure non-negativity

        # self.on_hand += repl_received
        # self.on_hand -= sales
        # assert (self.on_hand>=0).all(), f"OH is negative for (update function)"

        # Remove executed actions from pipeline
        self.pipeline_mask[execution_mask] = False
        adjusted_timestep = self.current_timestep + self.pipeline_start_idx

        # Add new action to pipeline only if RP value is 1
        for fc in range(self.num_fcs):
            if self.rp_arrays[fc, self.current_timestep] == 1:
                lt = self.sample_lead_time(fc)
                # empty_slot = (~self.pipeline_mask[fc]).nonzero(as_tuple=True)[0][0]
                self.action_pipeline[fc, adjusted_timestep] = torch.tensor(
                    [actions[fc].item(),self.current_timestep, lt, multiplier[fc].item()])
                self.pipeline_mask[fc, adjusted_timestep] = True
                # lt = self.sample_constrained_lead_time(fc, self.current_timestep)
                # print("Updates")
                # print(fc,lt)
                # self.action_pipeline[fc].append((actions[fc].item(), self.current_timestep, lt, multiplier[fc].item()))

            # else:
                #     # If RP is 0, add a zero action to maintain consistency
                #     self.action_pipeline[fc].append((0, self.current_timestep))

        self.current_timestep += 1

    @torch.no_grad()
    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the current state for all FCs."""
        states = []
        multipliers = []

        for fc in range(self.num_fcs):
            # rp = self.rp_arrays[fc]
            lt = self.expected_lt[fc]
            # print(f"expected {fc}: {lt}")

            oh_curr = self.on_hand[fc]  # sales have already been deducted. Updated for new timestep
            assert oh_curr.item() >= 0, 'OH is negative (get_state function)'

            # Replenishment should be received only if the LT for action in pipeline matches to today
            # Calculate replenishment received (if any) for this timestep
            # repl_received = sum(action*mult for action, timestamp,lt,mult in self.action_pipeline[fc] if timestamp + lt == self.current_timestep)

            execution_mask = (self.action_pipeline[fc, :, 1] + self.action_pipeline[fc, :,2] == self.current_timestep) & self.pipeline_mask[fc]
            repl_received = (self.action_pipeline[fc, :, 0] * self.action_pipeline[fc, :, 3] * execution_mask.float()).sum()

            # On-hands after replenishment
            oh_repl = oh_curr + repl_received

            # Agent needs to see how much is left in the action pipeline yet to replen before taking an action.
            # Done after current timestamp replen is removed
            # action_pipeline = sum(action*mult for action, _ ,_ ,mult in self.action_pipeline[fc])
            future_mask = (self.action_pipeline[fc, :, 1] + self.action_pipeline[fc, :, 2] >= self.current_timestep) & self.pipeline_mask[fc]
            action_pipeline = (self.action_pipeline[fc, :, 0] * self.action_pipeline[fc, :, 3] * future_mask.float()).sum()

            # next_lt = self.sample_lead_time(fc)  # Sample next LT for demand calculation
            dem_lt = self.forecast[fc, self.current_timestep:self.current_timestep + lt].sum()

            # Calculate demand during coverage period (CP) using the RP array
            cp_start = self.current_timestep + lt
            # cp_end = min(cp_start + torch.where(rp[cp_start:] == 1)[0][0].item() + 1, self.forecast_horizon)
            next_rp = torch.where(self.rp_arrays[fc, cp_start+1:] == 1)[0]
            if len(next_rp) > 0:
                # because we index it from cp_start onwards, the indices obtained from torch.where will include cp_start as 0, so add +1 there,
                # Another +1 is needed for the correct index to be used on the outer tensor
                # if next_rp[0][0] is 0 which means the current step itself, then we need to pick the next index so that 2 timesteps start and end are used
                cp_end = min(cp_start + 1 + next_rp[0].item(), self.forecast_horizon)
            else:
                cp_end = self.forecast_horizon

            dem_cp = self.forecast[fc, cp_start:cp_end].sum()

            # Realized sales for past lt time-steps
            # rs_lt = self.sales_history[fc, -lt:].sum()
            rs_avg = self.sales_history[fc, -(self.lbw):].mean()

            # Calculate days left till next decision
            next_decision = torch.where(self.rp_arrays[fc,self.current_timestep:] == 1)[0]
            days_left = next_decision[0].item() if len(next_decision) > 0 else 0

            # Get pending actions
            pending_actions = self.action_pipeline[fc, future_mask, 0] * self.action_pipeline[fc, future_mask, 3]
            # Pad or truncate to fixed size
            pending_actions_fixed = torch.zeros(self.max_pending_actions)
            pending_actions_fixed[:pending_actions.shape[0]] = pending_actions[:self.max_pending_actions] # 3 vs 5

            # act_hist = list(self.action_history[fc, -(self.lbw):])
            # Mask action history to include only actions in transit
            masked_act_hist = self.action_history[fc, -self.lbw:] * future_mask[-self.lbw:]
            rs_lt = list(self.sales_history[fc, -(self.lbw):])
            # dem_lt = list(self.forecast[fc, self.current_timestep:self.current_timestep + lt])
            fc_multiplier = int(rs_avg.item())  # self.forecast[fc, cp_start:cp_end].mean()
            # multip = list(self.multiplier_history[fc, -(self.lbw):])

            multip = torch.zeros(self.max_pending_actions)
            multip[:pending_actions.shape[0]] = self.action_pipeline[fc, future_mask, 3][:self.max_pending_actions]

            # previous state space - oh_curr, action_pipeline, repl_received, oh_repl, dem_cp, dem_lt, rs_lt , days_left
            #
            # print(f"FC {fc}: oh_curr = {oh_curr}, oh_repl = {oh_repl}, repl_received = {repl_received}, fc_multiplier = {fc_multiplier}, dem_lt:{dem_lt}, dem_cp:{dem_cp}")
            # fc_state = torch.tensor([oh_curr, fc_multiplier, repl_received, oh_repl, dem_cp, dem_lt] +act_hist+ multip+ rs_lt, dtype=torch.float32)
            fc_state = torch.cat([
                torch.tensor([oh_curr, fc_multiplier, repl_received, oh_repl, dem_cp, dem_lt]),
                pending_actions_fixed,
                multip,
                # masked_act_hist,
                torch.tensor(rs_lt),
            ])
            states.append(fc_state)
            multipliers.append(fc_multiplier)
        return torch.cat(states), torch.tensor(multipliers)  #states #torch.cat(states)

    def get_state_dim(self) -> int:
        """Return the dimension of the state vector."""
        base_features = 6  # oh_curr, fc_multiplier, repl_received, oh_repl, dem_cp, dem_lt, action_pipeline, days_left
        return (base_features + self.max_pending_actions*2 + self.lbw ) * self.num_fcs

        # return 12 * self.num_fcs

    def get_current_time_step(self):
        """
        Returns the time step of the state
        :return:
        """
        return self.current_timestep


def reset(seed,inventory_state):
    if reset_count>=1:
        inventory_state.set_seed(seed)
        inventory_state.reinitialize()

    # print(f"Forecast \n: {forecast}")
    # print(f"Mapped Demand \n: {mapped_demand}")
    inventory_state.set_forecast(forecast)

    return inventory_state.get_state()

def step(input,demand):
    state,multiplier = input
    inv_begin = state[0::fc_st_dim].reshape(-1, num_fcs)
    inv_repl = state[3::fc_st_dim].reshape(-1, num_fcs)
    repl_received = inv_repl - inv_begin

    # print("\n",inventory_state.action_pipeline)
    # print(f"inv_begin: {inv_begin}, inv_repl :{inv_repl}, repl:{repl_received}")
    sales = torch.min(inv_repl, demand) # random sales
    actions = torch.ceil(torch.rand(num_fcs) * 3)  # Random actions
    inventory_state.update(sales.flatten(), actions.flatten(),multiplier.flatten(),repl_received.flatten())

    next_state = inventory_state.get_state()
    # print(f"Next state: {next_state}, sales:{ sales.item()}, action :{actions.item()}, repl:{repl_received.item()}")
    return next_state

if __name__=="__main__":
    root = rootutils.setup_root(__file__, pythonpath=True)
    data_dir = root / "data/item_id_873764"
    # Initialize the StateSpace
    num_fcs = 3
    # ecdf = [(1, 0.1), (2, 0.3), (3, 0.6), (4, 0.9), (5, 1.0)]  # ECDF for FC 1  # Deterministic LT: [(0, 0.0), (1, 0.0), (2, 1.0)]
    with open(data_dir / 'lt_fc_ecdfs.pkl', 'rb') as f:
        fc_leadtimes = pickle.load(f)


    sel_fcs = [4270, 6284, 6753]
    lt_ecdfs = [fc_leadtimes.get(fc_id) for fc_id in sel_fcs]

    # lt_params = [(3, 2)] * num_fcs
    # lt_ecdfs = [ecdf] * num_fcs
    forecast_horizon = 100
    reset_count = 0
    # Create RP arrays (example: review every 3 days for all FCs)
    rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(forecast_horizon)] for _ in range(num_fcs)]
    # Set the forecast (normally provided by the environment)
    rng = torch.Generator().manual_seed(seed)
    # mapped_demand = torch.ceil(rng.rand((num_fcs, forecast_horizon),device = device) * 9)
    mapped_demand = torch.randint(0, 25, size=(num_fcs, forecast_horizon),generator=rng)
    forecast = torch.randint(0, 25, size=(num_fcs, forecast_horizon),generator=rng)

    # forecast = torch.ceil(torch.rand((num_fcs, forecast_horizon)) * 10)  # Random forecast for demonstration

    seed = 246
    # Initialize
    inventory_state = StateSpace(seed, num_fcs, lt_ecdfs, rp_arrays, forecast_horizon)
    # state = inventory_state.get_state()

    fc_st_dim = inventory_state.get_state_dim() // num_fcs
    # Simulate a few timesteps
    for i in range(100):
        print(i, seed)
        # print(inventory_state.action_pipeline)
        input = reset(seed, inventory_state)
        # print(f"Current state: {input}")

        for _ in range(forecast_horizon):
            # print(i, _, len(input[0]))
            next_state = step(input, mapped_demand[:, _])
            input = next_state
        # print("\n\n\n")
        reset_count += 1
        seed +=1

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
    #         # print("sales for