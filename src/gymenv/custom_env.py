import gym
from gym import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from src.data.components.state_space import StateSpace
import torch

class MultiSKUEnvironment(gym.Env):
    def __init__(self, env_cfg: Dict[str, Any]):
        super().__init__()
        self.item_nbr = env_cfg['item_nbr']
        self.data_dir = Path(env_cfg['data_dir'])
        self.reset_count = 0
        self.base_seed = env_cfg.get('seed', 42)
        self.load_data()
        self.initialize_parameters(env_cfg)
        self.setup_state_space()
        self.setup_action_space()

    def load_data(self):
        self.common_wks_list = sorted(pd.read_pickle(self.data_dir / 'common_wks_list.pkl'))
        self.node_ids_list = sorted(pd.read_pickle(self.data_dir / 'node_ids_list.pkl'))
        self.geo_ids_list = sorted(pd.read_pickle(self.data_dir / 'geo_ids_list.pkl'))
        self.benefits_data_table = pd.read_pickle(self.data_dir / 'benefits_data_table.pkl')
        self.demand_dist_data_table = pd.read_pickle(self.data_dir / 'demand_dist_data_table.pkl')
        self.ECDF_data_table = pd.read_pickle(self.data_dir / 'ECDF_data_table.pkl')
        self.LT_dist_data_table = pd.read_pickle(self.data_dir / 'LT_dist_data_table.pkl')
        self.pl_ratio_table = pd.read_pickle(self.data_dir / 'placement_ratio_tables.pkl')

    def initialize_parameters(self, env_cfg: Dict[str, Any]):
        self.hc = env_cfg.get('holding_cost', 0.1)
        self.sc = env_cfg.get('shortage_cost', 1.0)
        self.fc_lt_mean = np.array(env_cfg.get('fc_lt_mean', [5]))
        self.fc_lt_var = np.array(env_cfg.get('fc_lt_var', [1]))
        self.num_fc = env_cfg.get('num_fc', 1)
        self.num_sku = env_cfg.get('num_sku', 1)
        self.num_regions = len(self.geo_ids_list)
        self.forecast_horizon = env_cfg.get('forecast_horizon', 100)
        self.starting_week = env_cfg.get('starting_week', 12351)

    def setup_state_space(self):
        lead_times = np.random.normal(self.fc_lt_mean, self.fc_lt_var).astype(int).ravel()
        rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(self.forecast_horizon)] for _ in range(self.num_fc)]
        self.inventory_state = StateSpace(seed= self.base_seed, num_fcs=self.num_fc, lt_values=lead_times, rp_arrays=rp_arrays,
                                          forecast_horizon=self.forecast_horizon)
        obs_shape = (self.inventory_state.get_state_dim(),)
        self.observation_space = spaces.Box(low=0, high=700, shape=obs_shape, dtype=np.float32)

    def setup_action_space(self):
        # self.action_space = spaces.Box(low=0, high=20, shape=(self.num_fc, self.num_sku), dtype=np.float32)
        self.action_space = spaces.Discrete(20)

    def generate_demand_data(self):
        """ Can be removed"""
        dem = pd.DataFrame()
        dem['wm_yr_wk'] = np.array([[i] * 7 for i in self.common_wks_list]).flatten()
        for k in self.demand_dist_data_table.keys():
            temp = self.demand_dist_data_table[k][
                ['fcst_Sat', 'fcst_Sun', 'fcst_Mon', 'fcst_Tue', 'fcst_Wed', 'fcst_Thu', 'fcst_Fri']].values.flatten()
            dem[k] = temp
        return dem

    def gen_realised_dem(self, exp_dem):
        realised_dem = exp_dem.copy()
        for i, row in self.ECDF_data_table.iterrows():
            x, Fx = row['x'], row['Fx']
            num_samples = len(exp_dem)
            p = np.random.rand(num_samples)
            samples = np.interp(p, Fx, x)
            realised_dem.iloc[:, i] += samples
        return realised_dem.clip(lower=0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        if seed is not None:
            self.base_seed = seed

        self.reset_count += 1
        new_seed = self.base_seed + self.reset_count

        self.dem_ref = self.generate_demand_data()
        self.Exp_demand = self.generate_demand_data()
        self.start_time_index = self.Exp_demand[self.Exp_demand['wm_yr_wk'] == self.starting_week].index[0]
        self.Exp_demand = self.Exp_demand.iloc[self.start_time_index:, 1:]

        curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        mapped_forecast = np.array(self.Exp_demand.iloc[:self.forecast_horizon]) @ curr_pl_ratio
        mapped_forecast = mapped_forecast.iloc[:, :self.num_fc]

        # Someway to get timestamp in state to go back to zero
        # Update the seed and reinitialize the existing state space
        self.inventory_state.set_seed(new_seed)
        self.inventory_state.reinitialize()
        self.inventory_state.set_forecast(np.array(mapped_forecast.T))
        # set timestep = 0

        self.realized_demand_all = self.gen_realised_dem(self.Exp_demand)

        # self.state, self.multiplier = self.inventory_state.get_state()
        self.state = self.inventory_state.get_state()

        return self.state.numpy()

    def step(self, action):
        t = self.inventory_state.get_current_time_step()
        done = bool(t==self.inventory_state.endpoint)  # Define your terminal condition



        realized_demand = np.ceil(self.Exp_demand.iloc[t].values.reshape(-1, self.num_sku).astype(float))
        curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        fc_st_dim =self.inventory_state.get_state_dim()//self.num_fc
        inventory_at_beginning_of_day = self.state[0::fc_st_dim].reshape(-1, self.num_sku)
        inventory_after_replenishment = self.state[3::fc_st_dim].reshape(-1, self.num_sku)

        reward, sales_at_FC, holding_cost, shortage_cost, mapped_demand = self.calculate_reward(
            inventory_after_replenishment, realized_demand, curr_pl_ratio)

        repl_received = inventory_after_replenishment - inventory_at_beginning_of_day

        self.inventory_state.update(sales=torch.Tensor(sales_at_FC).flatten(), actions=torch.Tensor([action]).flatten())

        self.state = self.inventory_state.get_state()
        inventory_at_end_of_day = inventory_after_replenishment - sales_at_FC

        info = {
            'inv_at_beginning_of_day': inventory_at_beginning_of_day.item(),
            'inv_after_replen': inventory_after_replenishment.item(),
            'realised_dem': realized_demand,
            'inv_at_end_of_day': inventory_at_end_of_day.item(),
            'sales_at_FC': sales_at_FC.item(),
            'holding_cost': holding_cost.item(),
            'shortage_cost': shortage_cost.item(),
            # 'repl_ord': action * self.multiplier,
            'action': action,
            # 'multiplier': self.multiplier,
            'mapped_dem': mapped_demand.item(),
            'repl_rec': repl_received.item()
        }

        return self.state.numpy(), reward, done, info

    def calculate_reward(self, inventory, demand, pl_ratio):
        # This might need to change based on demand granularity(FC/Geo)
        #
        inventory = inventory.numpy()
        mapped_demand = np.dot(demand.T,pl_ratio)
        mapped_demand = mapped_demand[:,:self.num_fc]

        sales = np.minimum(inventory, mapped_demand)
        holding_cost = self.hc * np.maximum(inventory - mapped_demand, 0)
        shortage_cost = self.sc * np.maximum(-(inventory - mapped_demand),0)
        sales_revenue = self.sc * sales * 5
        reward = np.sum(sales_revenue) - np.sum(holding_cost) - np.sum(shortage_cost)
        return reward, sales, holding_cost, shortage_cost, mapped_demand


