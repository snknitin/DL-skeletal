import numpy as np
import torch
from collections import deque
import gym
from gym import spaces
import timeit
import yaml
import pandas as pd
# from state_space_wo_explicit_action import *
# from src.data.components.SS_with_act_factor import *
from src.data.components.SS_with_act_fact import StateSpace


# # Set a fixed seed
# seed = 20
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

class MultiSKUEnvironment(gym.Env):
    def __init__(self, config_path, fc_lt_mean, fc_lt_var, reward_func, starting_week=12351):
        super(MultiSKUEnvironment, self).__init__()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        common_wks_list = sorted(pd.read_pickle(self.config['common_wks_list']))
        node_ids_list = sorted(pd.read_pickle(self.config['node_ids_list']))
        geo_ids_list = sorted(pd.read_pickle(self.config['geo_ids_list']))
        benefits_data_table = pd.read_pickle(self.config['benefits_data_table'])
        demand_dist_data_table = pd.read_pickle(self.config['demand_dist_data_table'])
        self.ECDF_data_table = pd.read_pickle(self.config['ECDF_data_table'])
        LT_dist_data_table = pd.read_pickle(self.config['LT_dist_data_table'])
        self.pl_ratio_table = pd.read_pickle(self.config['Placement_Ratio_table'])

        self.hc = self.config['holding_cost']
        self.sc = self.config['shortage_cost']

        self.fc_lt_mean = np.array(fc_lt_mean)
        self.fc_lt_var = np.array(fc_lt_var)

        self.num_fc = 1  # len(node_ids_list)
        self.num_sku = 1
        self.num_regions = len(geo_ids_list)

        self.epoch = 0

        def generate_demand_data(demand_table):
            dem = pd.DataFrame()
            dem['wm_yr_wk'] = np.array([[i] * 7 for i in common_wks_list]).flatten()
            for k in demand_table.keys():
                temp = demand_table[k][['fcst_Sat', 'fcst_Sun', 'fcst_Mon', 'fcst_Tue', 'fcst_Wed', 'fcst_Thu',
                                        'fcst_Fri']].values.flatten()
                dem[k] = temp
            return dem

        self.dem_ref = generate_demand_data(demand_dist_data_table)
        self.Exp_demand = generate_demand_data(demand_dist_data_table)
        self.starting_week = starting_week  # 12351 -> 15th-Jan
        self.start_time_index = self.Exp_demand[self.Exp_demand['wm_yr_wk'] == self.starting_week].index[0]
        self.Exp_demand = self.Exp_demand.iloc[self.start_time_index:, 1:]

        # Reward function
        self.reward_func = reward_func

        # Initialise state space
        self.forecast_horizon = 100
        lead_times = np.random.normal(self.fc_lt_mean, self.fc_lt_var).astype(int).ravel()
        rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(self.forecast_horizon)] for _ in range(self.num_fc)]
        self.State_Space = StateSpace(num_fcs=self.num_fc, lt_values=lead_times, rp_arrays=rp_arrays,
                                      forecast_horizon=self.forecast_horizon)

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=3, shape=(self.num_fc, self.num_sku),
                                       dtype=np.float32)  # Define action space based on the problem
        obs_shape = (self.State_Space.get_state_dim(),)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=obs_shape, dtype=np.float32)

    def gen_realised_dem(self, exp_dem, ecdf_data):
        realised_dem = exp_dem.copy()
        for i, row in ecdf_data.iterrows():
            x = row['x']
            Fx = row['Fx']
            # Generate a random number between 0 and 1
            num_samples = len(exp_dem)
            p = np.random.rand(num_samples)
            # Generate a sample by interpolating the empirical CDF
            samples = np.interp(p, Fx, x)
            realised_dem.iloc[:, i] += samples
        realised_dem = realised_dem.clip(lower=0)
        return realised_dem

    def reset(self):

        curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        mapped_forecast = np.array(self.Exp_demand.iloc[:self.forecast_horizon]) @ curr_pl_ratio
        mapped_forecast = mapped_forecast.iloc[:, :self.num_fc]  # For 1 FC

        self.temp_mapped_forecast = mapped_forecast.copy()

        lead_times = np.random.normal(self.fc_lt_mean, self.fc_lt_var).astype(int).ravel()
        rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(self.forecast_horizon)] for _ in range(self.num_fc)]
        self.State_Space = StateSpace(num_fcs=self.num_fc, lt_values=lead_times, rp_arrays=rp_arrays,
                                      forecast_horizon=self.forecast_horizon)

        self.State_Space.set_forecast(np.array(mapped_forecast.T))

        self.realized_demand_all = self.gen_realised_dem(self.Exp_demand, self.ECDF_data_table)

        self.state, self.multiplier = self.State_Space.get_state()

        return self.state

    def step(self, action, t):
        # action = np.array([1])
        # Sample realized demand and update inventory
        # realized_demand = self.realized_demand_all.iloc[t].values.reshape(-1, self.num_sku).astype(float)
        realized_demand = np.ceil(
            self.Exp_demand.iloc[t].values.reshape(-1, self.num_sku).astype(float))  # Without Stochasticity
        curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]  # Removing t for simulation
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        index = torch.tensor([(i * self.State_Space.get_state_dim()) + 0 for i in range(self.num_fc)])
        inventory_at_beginning_of_day = self.state[index].numpy().reshape(-1, self.num_sku)
        index = torch.tensor([(i * self.State_Space.get_state_dim()) + 3 for i in range(self.num_fc)])
        inventory_after_replenishment = self.state[index].numpy().reshape(-1, self.num_sku)
        # index = torch.tensor([(i*self.State_Space.get_state_dim())+7 for i in range(self.num_fc)]) #Check the logic for multiple FCs again
        # multiplier = self.state[index:].numpy().reshape(-1,self.num_sku).mean(axis=0)                    #Check the logic for multiple FCs again
        multiplier = self.multiplier.copy()

        # Obtain updated inventory, reward
        reward, sales_at_FC, holding_cost, shortage_cost, mapped_demand = self.reward_func(
            inventory_after_replenishment, realized_demand, curr_pl_ratio, self.hc, self.sc)

        repl_received = inventory_after_replenishment - inventory_at_beginning_of_day

        # Computing Proh_Oh only for plotting======================
        # ratio_to_plot = []
        # for fc in range(self.num_fc):
        #         ratio_to_plot.append(self.order_qty[fc])
        # ==========================================================

        self.State_Space.update(sales=torch.from_numpy(sales_at_FC.flatten()),
                                actions=torch.from_numpy(action.flatten()),
                                multiplier=torch.from_numpy(multiplier.flatten()))

        # Get new state
        self.state, self.multiplier = self.State_Space.get_state()
        inventory_at_end_of_day = inventory_after_replenishment - sales_at_FC

        done = False  # Define your terminal condition
        info = {'inv_at_beginning_of_day': inventory_at_beginning_of_day,
                'inv_after_replen': inventory_after_replenishment,
                'realised_dem': realized_demand,
                'inv_at_end_of_day': inventory_at_end_of_day,
                'sales_at_FC': sales_at_FC,
                'holding_cost': holding_cost,
                'shortage_cost': shortage_cost,
                'repl_ord': action * multiplier,
                'action': action,
                'multiplier': multiplier,
                'mapped_dem': mapped_demand,
                'repl_rec': repl_received}
        # 'ratio_to_plot': ratio_to_plot}

        return self.state, reward, done, info