import pickle

import gym
from gym import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from src.data.components.state_space import StateSpace
import torch
import ot

ot.utils.check_random_state(seed=42)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()


class MultiFCEnvironment(gym.Env):
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
        self.cost_data = pd.read_pickle(
            self.data_dir / 'cost_data_table.pkl')  # Computed using Min-Max over benefits data table
        with open(self.data_dir / 'lt_fc_ecdfs.pkl', 'rb') as f:
            self.fc_leadtimes = pickle.load(f)
        self.sales_table = pd.read_pickle(self.data_dir / 'Prod_Sales_tables.pkl')

    def initialize_parameters(self, env_cfg: Dict[str, Any]):
        self.hc = env_cfg.get('holding_cost', 0.1)
        self.sc = env_cfg.get('shortage_cost', 1.0)
        self.fc_lt_mean = np.array(env_cfg.get('fc_lt_mean', [5]))
        self.fc_lt_var = np.array(env_cfg.get('fc_lt_var', [1]))
        # self.lt_ecdfs = env_cfg.get('lt_ecdfs', [(1, 0.1), (2, 0.3), (3, 0.6), (4, 0.9), (5, 1.0)])
        self.num_fcs = env_cfg.get('num_fcs', 16)
        self.num_sku = env_cfg.get('num_sku', 1)
        self.num_regions = len(self.geo_ids_list)
        self.forecast_horizon = env_cfg.get('forecast_horizon', 100)
        self.starting_week = env_cfg.get('starting_week', 12351)
        self.sel_fcs = env_cfg.get('sel_FCs', [4270, 6284, 6753])

    def setup_state_space(self):
        sel_fcs = [int(fc_id) for fc_id in self.sel_fcs]
        self.sel_fcs = sel_fcs
        lt_ecdfs = [self.fc_leadtimes.get(fc_id) for fc_id in sel_fcs]
        rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(self.forecast_horizon)] for _ in range(self.num_fcs)]
        self.inventory_state = StateSpace(seed=self.base_seed, num_fcs=self.num_fcs, lt_ecdfs=lt_ecdfs,
                                          rp_arrays=rp_arrays,
                                          forecast_horizon=self.forecast_horizon)
        obs_shape = (self.inventory_state.get_state_dim(),)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=obs_shape, dtype=np.float32)

    def setup_action_space(self):
        # self.action_space = spaces.Box(low=0, high=20, shape=(self.num_fc, self.num_sku), dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([20] * self.num_fcs)
        self.action_space = spaces.Box(low=0, high=3, shape=(self.num_fcs, self.num_sku),
                                       dtype=np.float32)  # Define action space based on the problem

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

    def generate_act_sales_data(self):
        sales = pd.DataFrame()
        sales['wm_yr_wk'] = np.array([[i] * 7 for i in self.common_wks_list]).flatten()
        for k in self.sales_table.keys():
            temp = self.sales_table[k][
                ['SFFC_Sat', 'SFFC_Sun', 'SFFC_Mon', 'SFFC_Tue', 'SFFC_Wed', 'SFFC_Thu', 'SFFC_Fri']].values.flatten()
            sales[k] = temp
        return sales

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        if seed is not None:
            self.base_seed = seed
        new_seed = self.base_seed + self.reset_count

        self.dem_ref = self.generate_demand_data()
        self.Exp_demand = self.generate_demand_data()
        self.start_time_index = self.Exp_demand[self.Exp_demand['wm_yr_wk'] == self.starting_week].index[0]
        self.Exp_demand = self.Exp_demand.iloc[self.start_time_index:, 1:]
        self.prod_sales = self.generate_act_sales_data()

        curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        # Someway to get timestamp in state to go back to zero
        # Update the seed and reinitialize the existing state space
        if self.reset_count >= 1:
            self.inventory_state.set_seed(new_seed)
            self.inventory_state.reinitialize()

        curr_pl_ratio = curr_pl_ratio[self.sel_fcs]
        curr_pl_ratio = curr_pl_ratio[curr_pl_ratio[curr_pl_ratio != 0].any(axis=1)]
        sel_regions = [f'geo_id_{i}' for i in curr_pl_ratio.index]

        # Initialize options as an empty dictionary if None is provided
        if options is None:
            options = {}

        # Check if 'test' is in options and if it's True
        if options.get('test', False):  # Default to False if 'test' is not present
            # # For testing only
            mean_noise, std_dev_noise = 0, 0
            self.Exp_demand = (
                        self.Exp_demand + np.random.normal(mean_noise, std_dev_noise, self.Exp_demand.shape)).clip(
                lower=0)  # Add random noise
            self.realized_demand_all = self.prod_sales[sel_regions]
        else:
            ## For selecting regions mapped to selected FCs==============================================================
            self.realized_demand_all = np.ceil(self.gen_realised_dem(self.Exp_demand))

        self.Exp_demand_modi = self.Exp_demand[sel_regions]
        self.realized_demand_all = self.realized_demand_all[sel_regions]
        mapped_forecast = np.array(self.Exp_demand_modi.iloc[:self.forecast_horizon]) @ curr_pl_ratio
        self.benefits_data_table = self.benefits_data_table.loc[self.sel_fcs, [str(i) for i in curr_pl_ratio.index]]
        ##  =========================================================================================================

        self.inventory_state.set_forecast(np.array(mapped_forecast.T))

        # self.state, self.multiplier = self.inventory_state.get_state()
        self.state, self.multiplier = self.inventory_state.get_state()
        self.reset_count += 1
        return self.state.numpy()

    def step(self, action):
        t = self.inventory_state.get_current_time_step()
        done = bool(t == self.inventory_state.endpoint)  # Define your terminal condition

        # realized_demand = np.ceil(self.Exp_demand.iloc[t].values.reshape(-1, self.num_sku).astype(float))
        # curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]
        # curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        ##------------------------------------------------------
        ## Sample Realised Demand from selected Regions (According to PR)
        curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]  # Removing t for simulation
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]
        curr_pl_ratio = curr_pl_ratio[self.sel_fcs]  # 4270, 6284, 6753,
        curr_pl_ratio = curr_pl_ratio[curr_pl_ratio[curr_pl_ratio != 0].any(axis=1)]
        realized_demand = self.realized_demand_all.iloc[t].values.reshape(-1, self.num_sku).astype(float)
        # realized_demand = np.ceil(self.Exp_demand_modi.iloc[t].values.reshape(-1, self.num_sku).astype(float)) #Without Stochasticity
        ##------------------------------------------------------

        fc_st_dim = self.inventory_state.get_state_dim() // self.num_fcs
        inventory_at_beginning_of_day = self.state[0::fc_st_dim].reshape(-1, self.num_fcs)
        inventory_after_replenishment = self.state[3::fc_st_dim].reshape(-1, self.num_fcs)

        # reward, sales_at_FC, holding_cost, shortage_cost, mapped_demand = self.calculate_reward(
        #     inventory_after_replenishment, realized_demand, curr_pl_ratio)

        mapped_demand_pr = np.array(realized_demand.reshape(1, -1) @ curr_pl_ratio).flatten()
        # mapped_demand_pr = np.ceil(mapped_demand_pr)

        multiplier = self.multiplier.copy().reshape(self.num_fcs, -1)

        reward, sales_at_FC, holding_cost_pr, shortage_cost_pr, mapped_dem_ot, holding_qty_pr, shortage_qty_pr, holding_qty_ot, shortage_qty_ot = self.calculate_reward_OT(
            inventory_after_replenishment, realized_demand, mapped_demand_pr, self.benefits_data_table)

        assert mapped_demand_pr.sum() >= mapped_dem_ot.sum(), "f{t}, PR Not Equal to OT"

        repl_received = inventory_after_replenishment - inventory_at_beginning_of_day
        inventory_at_end_of_day = inventory_after_replenishment - sales_at_FC

        self.inventory_state.update(sales=torch.from_numpy(sales_at_FC).flatten(),
                                    actions=torch.Tensor(np.array(action)).flatten(),
                                    multiplier=torch.from_numpy(multiplier.flatten()))
        self.state, self.multiplier = self.inventory_state.get_state()

        info = {
            'inv_at_beginning_of_day': inventory_at_beginning_of_day.flatten(),
            'inv_after_replen': inventory_after_replenishment.flatten(),
            'realised_dem': realized_demand.flatten(),
            'inv_at_end_of_day': inventory_at_end_of_day.flatten(),
            'sales_at_FC': sales_at_FC.flatten(),
            # 'holding_cost': holding_cost.flatten(),
            # 'shortage_cost': shortage_cost.flatten(),
            'holding_cost': holding_cost_pr.flatten(),
            'shortage_cost': shortage_cost_pr.flatten(),
            'repl_ord': (action * multiplier).flatten(),
            'action': action.flatten(),
            'multiplier': multiplier.flatten(),
            'mapped_dem': mapped_demand_pr.flatten(),
            'repl_rec': repl_received.flatten()
        }

        return self.state.numpy(), reward, done, info

    def calculate_reward(self, inventory, demand, pl_ratio):
        # This might need to change based on demand granularity(FC/Geo)
        #
        inventory = inventory.numpy()
        mapped_demand = np.dot(demand.T, pl_ratio)
        mapped_demand = mapped_demand[:, :self.num_fcs]

        sales = np.minimum(inventory, mapped_demand)
        holding_cost = self.hc * np.maximum(inventory - mapped_demand, 0)
        shortage_cost = self.sc * np.maximum(-(inventory - mapped_demand), 0)
        sales_revenue = self.sc * sales * 5
        # Calculate reward for each FC
        reward = sales_revenue - holding_cost - shortage_cost
        # Sum across the first axis to get a reward per FC

        ## Vector Reward =========================================
        reward = np.sum(reward, axis=0)  # Shape: (num_fcs,)

        ## Scalar Reward =========================================
        # reward = np.sum(reward)

        return reward, sales, holding_cost, shortage_cost, mapped_demand

    def calculate_reward_OT(self, inventory, realized_demand, mapped_dem_pr, benefits_data_table):

        inventory = inventory.numpy().reshape(-1, 1)

        temp_realized_demand = np.vstack((realized_demand, np.zeros((1, self.num_sku)))).astype(
            int)  # Add dummy demand row
        temp_inventory = np.vstack((inventory, np.zeros((1, self.num_sku)))).astype(int)  # Add dummy supply row

        # Calculate the column sums for demands and inventories excluding the dummy rows
        demand_sums = np.sum(temp_realized_demand[:-1, :], axis=0)
        inventory_sums = np.sum(temp_inventory[:-1, :], axis=0)

        # Calculate the differences for each SKU
        differences = demand_sums - inventory_sums

        # Adjust the dummy rows to balance the sums for each SKU
        for i in range(len(differences)):
            if differences[i] > 0:
                temp_inventory[-1, i] = differences[i]
            elif differences[i] < 0:
                temp_realized_demand[-1, i] = -differences[i]

        reward = 0
        holding_cost_ot, shortage_cost_ot = 0, 0
        holding_cost_pr, shortage_cost_pr = 0, 0

        for i in range(temp_inventory.shape[1]):
            M = np.array(benefits_data_table)
            M = 1 - min_max_scaler.fit_transform(np.array(benefits_data_table.T)).T

            benefits = 1 - M

            M = np.vstack((M, np.ones((1, M.shape[1])) * self.sc * 5))
            M = np.hstack((M, np.ones((M.shape[0], 1)) * self.hc * 5))

            supply = temp_inventory[:, i]
            demand = temp_realized_demand[:, i]

            uot = ot.solve(M / M.max(), supply.astype('float'), demand.astype('float'),
                           method='emd')  # method='sinkhorn')

            sales_at_FC = uot.plan[:-1, :-1].sum(axis=1)

            # Calculate Mapped Demand
            mapped_dem = uot.plan[:-1, :-1].sum(axis=1).astype('int')

            ## (OT) Calculate reward
            holding_cost_ot += max(0, sum(temp_inventory[:-1, i])) * self.hc
            holding_qty_ot = -min(0, differences[i])
            shortage_cost_ot += max(0, differences[i]) * self.sc
            shortage_qty_ot = max(0, differences[i])

            ## (PR) Calculate reward
            diff = mapped_dem_pr.flatten() - inventory.flatten()
            holding_cost_pr = np.where(diff > 0, 0, diff * -self.hc * 5) / M.max()
            holding_qty_pr = sum(np.where(diff > 0, 0, -diff))
            shortage_cost_pr = np.where(diff > 0, diff * self.sc * 5, 0) / M.max()
            shortage_qty_pr = sum(np.where(diff > 0, diff, 0))

            local_cost = sum(shortage_cost_pr + holding_cost_pr)

            global_cost = uot.value

            reward = -1 * (0 * global_cost + 1 * local_cost)

        return reward, sales_at_FC, holding_cost_pr, shortage_cost_pr, mapped_dem, holding_qty_pr, shortage_qty_pr, holding_qty_ot, shortage_qty_ot