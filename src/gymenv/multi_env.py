import pickle
import cvxpy as cp
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



class MultiFCEnvironment(gym.Env):
    def __init__(self, env_cfg: Dict[str, Any]):
        super().__init__()

        self.item_nbr = env_cfg['item_nbr']
        self.data_dir = Path(env_cfg['data_dir'])
        self.reset_count = 0
        self.base_seed = env_cfg.get('seed', 42)
        self.initialize_parameters(env_cfg)
        self.load_data()
        self.setup_state_space()
        self.setup_action_space()

    def generate_demand_data(self):
        """ Can be removed"""
        dem = pd.DataFrame()
        dem['wm_yr_wk'] = np.array([[i] * 7 for i in self.common_wks_list]).flatten()
        for k in self.demand_dist_data_table.keys():
            temp = self.demand_dist_data_table[k][
                ['fcst_Sat', 'fcst_Sun', 'fcst_Mon', 'fcst_Tue', 'fcst_Wed', 'fcst_Thu', 'fcst_Fri']].values.flatten()
            dem[k] = temp
        return dem

    def gen_forecasterror(self):
        num_samples = 100000
        expected_forecast_error = dict()
        for i, row in self.ECDF_data_table.iterrows():
            geo_id = 'geo_id_' + str(row['geo_id'])
            x, Fx = row['x'], row['Fx']
            p = np.random.rand(num_samples)
            samples = np.interp(p, Fx, x)
            expected_forecast_error[geo_id] = [np.mean(samples)]
        return pd.DataFrame.from_dict(expected_forecast_error, orient='columns')

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

    def load_data(self):
        self.common_wks_list = sorted(pd.read_pickle(self.data_dir / 'common_wks_list.pkl'))
        self.node_ids_list = sorted(pd.read_pickle(self.data_dir / 'node_ids_list.pkl'))
        self.geo_ids_list = sorted(pd.read_pickle(self.data_dir / 'geo_ids_list.pkl'))
        self.benefits_data_table = pd.read_pickle(self.data_dir / 'benefits_data_table.pkl')
        self.demand_dist_data_table = pd.read_pickle(self.data_dir / 'demand_dist_data_table.pkl')
        self.ECDF_data_table = pd.read_pickle(self.data_dir / 'ECDF_data_table.pkl')
        self.LT_dist_data_table = pd.read_pickle(self.data_dir / 'LT_dist_data_table.pkl')
        self.pl_ratio_table = pd.read_pickle(self.data_dir / 'placement_ratio_tables.pkl')
        # self.cost_data = pd.read_pickle(self.data_dir / 'cost_data_table.pkl') #Computed using Min-Max over benefits data table
        with open(self.data_dir / 'lt_fc_ecdfs.pkl', 'rb') as f:
            self.fc_leadtimes = pickle.load(f)
        self.sales_table = pd.read_pickle(self.data_dir / 'Prod_Sales_tables.pkl')
        self.dem_ref = self.generate_demand_data()  # (266,59) wm_wk, 58 geos
        self.Exp_demand = self.generate_demand_data()
        self.prod_sales = self.generate_act_sales_data()  # (266,59) wm_wk, 58 geos
        self.start_time_index = self.Exp_demand[self.Exp_demand['wm_yr_wk'] == self.starting_week].index[0]  # 119
        self.test_time_index = self.prod_sales[self.prod_sales['wm_yr_wk'] == self.test_week].index[0] # 203

        ## Sample Realised Demand from selected Regions (According to PR)
        #curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]  # 12351
        #Karthik - Hardcoding the curr_week_no so that we use the placement ratios obtained
        #from this week no for both training and evaluation. Need to be fixed
        curr_week_no = 12351
        curr_pl_ratio = self.pl_ratio_table[curr_week_no]  # (58,16) geo to fc
        curr_pl_ratio = curr_pl_ratio[self.sel_fcs]  # (58,num_fcs = 3)
        self.curr_pl_ratio = curr_pl_ratio[curr_pl_ratio[curr_pl_ratio != 0].any(axis=1)]  # (14,3) non zero geos

        all_zero_columns = self.curr_pl_ratio.columns[self.curr_pl_ratio.eq(0).all()]
        assert all_zero_columns.empty, f"Columns with 0 Placement Ratios: {list(all_zero_columns)}"

        self.sel_regions = [f'geo_id_{i}' for i in self.curr_pl_ratio.index]  # 14 geos
        self.exp_demand = self.Exp_demand.iloc[self.start_time_index:, 1:]  # (147,58)
        self.exp_dem_sel = self.exp_demand[self.sel_regions]  # (147,14)

        self.realized_demand_all = np.ceil(self.gen_realised_dem(self.exp_demand))  # (147,58)
        self.realized_demand_sel = self.realized_demand_all[self.sel_regions]  # (147,14)

        # For some reason ?
        self.benefits_data_table = self.benefits_data_table.loc[
            self.sel_fcs, [str(i) for i in self.curr_pl_ratio.index]]
        #Karthik - Moving the forecast error computation here from reset()
        self.mean_forecast_error = self.gen_forecasterror()

    def initialize_parameters(self, env_cfg: Dict[str, Any]):
        self.hc = env_cfg.get('holding_cost', 0.1)
        self.sc = env_cfg.get('shortage_cost', 1.0)

        self.fc_lt_mean = np.array(env_cfg.get('fc_lt_mean', [5]))
        self.fc_lt_var = np.array(env_cfg.get('fc_lt_var', [1]))
        # self.lt_ecdfs = env_cfg.get('lt_ecdfs', [(1, 0.1), (2, 0.3), (3, 0.6), (4, 0.9), (5, 1.0)])
        self.num_fcs = env_cfg.get('num_fcs', 16)
        self.num_sku = env_cfg.get('num_sku', 1)
        # self.num_regions = len(self.geo_ids_list)
        self.forecast_horizon = env_cfg.get('forecast_horizon', 100)
        self.starting_week = env_cfg.get('starting_week', 12351)
        self.test_week = env_cfg.get('test_week', 12411)

        self.sel_fcs = env_cfg.get('sel_FCs', [4270, 6284, 6753])
        sel_fcs = [int(fc_id) for fc_id in self.sel_fcs]
        self.sel_fcs = sel_fcs
        self.min_max_scaler = MinMaxScaler()
        self.compute_sla = False


    def setup_state_space(self):
        lt_ecdfs = [self.fc_leadtimes.get(fc_id) for fc_id in self.sel_fcs]
        rp_arrays = [[1 if (i + 1) % 1 == 0 else 0 for i in range(self.forecast_horizon)] for _ in range(self.num_fcs)]
        self.inventory_state = StateSpace(seed=self.base_seed, num_fcs=self.num_fcs, lt_ecdfs=lt_ecdfs,
                                          rp_arrays=rp_arrays,
                                          forecast_horizon=self.forecast_horizon)
        obs_shape = (self.inventory_state.get_state_dim(),)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=obs_shape, dtype=np.float32)

    def setup_action_space(self):
        # self.action_space = spaces.Box(low=0, high=20, shape=(self.num_fc, self.num_sku), dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([20] * self.num_fcs)
        self.action_space = spaces.Box(low=0, high=5, shape=(self.num_fcs, self.num_sku),
                                       dtype=np.float32)  # Define action space based on the problem - high is getting set manually

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Initialize options as an empty dictionary if None is provided
        if options is None:
            options = {}

        if seed is not None:
            self.base_seed = seed

        new_seed = self.base_seed + self.reset_count
        torch.manual_seed(new_seed)
        # np.random.seed(new_seed)

        # Someway to get timestamp in state to go back to zero
        # Update the seed and reinitialize the existing state space
        if self.reset_count >= 1:
            self.inventory_state.set_seed(new_seed)
            self.inventory_state.reinitialize()



        # Check if 'test' is in options and if it's True
        if options.get('actual_test', False):  # Default to False if 'test' is not present
            self.compute_sla = True
            self.realized_demand_all = self.prod_sales[self.sel_regions][self.test_time_index:]
            self.realized_demand_sel = self.realized_demand_all[self.sel_regions]  # (147,14)

            mapped_forecast = np.array(self.exp_dem_sel.iloc[:self.forecast_horizon]) @ self.curr_pl_ratio
            mapped_forecast = np.vstack([mapped_forecast.values, np.tile(mapped_forecast.values[-1, np.newaxis],
                                     (self.forecast_horizon - mapped_forecast.shape[0], 1))]) if mapped_forecast.shape[0] < self.forecast_horizon else mapped_forecast.values
            self.inventory_state.set_forecast(torch.from_numpy(mapped_forecast.T))

        elif options.get('simulated_test', False):
            self.compute_sla = True
            # Add random noise
            mean_noise, std_dev_noise = options['simulated_test']
            self.exp_demand = (
                        self.exp_demand + np.random.normal(mean_noise, std_dev_noise, self.exp_demand.shape)).clip(
                lower=0)
            self.realized_demand_all = np.ceil(self.gen_realised_dem(self.exp_demand))
            self.realized_demand_sel = self.realized_demand_all[self.sel_regions]  # (147,14)

            mapped_forecast = np.array(self.exp_dem_sel.iloc[:self.forecast_horizon]) @ self.curr_pl_ratio
            mapped_forecast = np.vstack([mapped_forecast.values, np.tile(mapped_forecast.values[-1, np.newaxis],
                                                                         (self.forecast_horizon - mapped_forecast.shape[
                                                                             0], 1))]) if mapped_forecast.shape[
                                                                                              0] < self.forecast_horizon else mapped_forecast.values
            self.inventory_state.set_forecast(torch.from_numpy(mapped_forecast.T))
        else:
            ## For selecting regions mapped to selected FCs==============================================================
            self.realized_demand_all = np.ceil(self.gen_realised_dem(self.exp_demand))
            self.realized_demand_sel = self.realized_demand_all[self.sel_regions]  # (147,14)

            mapped_forecast = np.array(self.exp_dem_sel.iloc[:self.forecast_horizon]) @ self.curr_pl_ratio
            self.inventory_state.set_forecast(torch.from_numpy(mapped_forecast.values.T))
            ##  =========================================================================================================
        #self.mean_forecast_error = self.gen_forecasterror()
        self.state, self.multiplier = self.inventory_state.get_state()
        self.reset_count += 1
        return self.state

    @torch.no_grad()
    def step(self, action):
        t = self.inventory_state.get_current_time_step()
        done = bool(t == self.inventory_state.endpoint)  # Define your terminal condition
        # realized_demand = np.ceil(self.Exp_demand.iloc[t].values.reshape(-1, self.num_sku).astype(float))
        # curr_week_no = self.dem_ref.iloc[self.start_time_index, 0]
        # curr_pl_ratio = self.pl_ratio_table[curr_week_no]

        realized_demand = self.realized_demand_sel.iloc[t].values.reshape(-1, self.num_sku).astype(float)
        # realized_demand = np.ceil(self.Exp_demand_modi.iloc[t].values.reshape(-1, self.num_sku).astype(float)) #Without Stochasticity

        fc_st_dim = self.inventory_state.get_state_dim() // self.num_fcs
        inventory_at_beginning_of_day = self.state[0::fc_st_dim].reshape(-1, self.num_fcs)
        inventory_after_replenishment = self.state[3::fc_st_dim].reshape(-1, self.num_fcs)



        mapped_demand_pr = np.array(realized_demand.reshape(1, -1) @ self.curr_pl_ratio).flatten()
        # mapped_demand_pr = np.ceil(mapped_demand_pr)

        ### Unbiased multiplier
        mean_forecast_error = (np.array(self.mean_forecast_error[self.sel_regions]) @ self.curr_pl_ratio)
        temp1 = self.multiplier.clone().reshape(self.num_fcs, -1)
        temp2 = mean_forecast_error
        temp2 = np.array(temp2).reshape(self.num_fcs, -1)
        multiplier = temp1 + temp2

        safety_stock = action.flatten() * multiplier.flatten()
        dem_cp_fc = self.state[1::fc_st_dim].reshape(-1, self.num_fcs)
        proj_oh_lt_fc = self.state[8::fc_st_dim].reshape(-1, self.num_fcs)
        # repl_ordered = torch.max(torch.tensor(0), safety_stock + dem_cp_fc - proj_oh_lt_fc)     #SS_Model -> Need to make changes in State space file as well while using this
        repl_ordered = action.flatten() * multiplier.flatten()                                  #Repl_Model

        reward, sales_at_FC, holding_cost_pr, shortage_cost_pr, mapped_dem_ot, holding_qty_pr, shortage_qty_pr, holding_qty_ot, shortage_qty_ot, uot_plan, x = self.calculate_reward_OT(
            inventory_after_replenishment, realized_demand, mapped_demand_pr, self.benefits_data_table, self.compute_sla)

        # reward, sales_at_FC, holding_cost_pr, shortage_cost_pr, mapped_demand_pr = self.calculate_reward(
        #     inventory_after_replenishment, realized_demand, mapped_demand_pr)

        # assert mapped_demand_pr.sum() >= mapped_dem_ot.sum(), "f{t}, PR Not Equal to OT"

        repl_received = inventory_after_replenishment - inventory_at_beginning_of_day
        inventory_at_end_of_day = inventory_after_replenishment - sales_at_FC

        self.inventory_state.update(sales=sales_at_FC.flatten(),
                                    actions=action.flatten(),
                                    multiplier=multiplier.flatten(),
                                    repl_received= repl_received.flatten(),
                                    repl_ordered=repl_ordered.flatten())  # Included repl_ordered in update

        self.state, self.multiplier = self.inventory_state.get_state()

        self.mapped_forecast_plotting = (np.array(self.exp_dem_sel.iloc[:self.forecast_horizon]) @ self.curr_pl_ratio).iloc[t,:].values  # Only for plotting, can be removed later
        info = {
            'inv_at_beginning_of_day': inventory_at_beginning_of_day.flatten(),
            'inv_after_replen': inventory_after_replenishment.flatten(),
            'realised_dem': realized_demand.flatten(),
            'inv_at_end_of_day': inventory_at_end_of_day.flatten(),
            'sales_at_FC': sales_at_FC.flatten(),
            'holding_cost': holding_cost_pr.flatten(),
            'shortage_cost': shortage_cost_pr.flatten(),
            'holding_qty_pr': holding_qty_pr.flatten(),
            'shortage_qty_pr': shortage_qty_pr.flatten(),
            'holding_qty_ot': holding_qty_ot.flatten(),
            'shortage_qty_ot': shortage_qty_ot.flatten(),
            # 'repl_ord': action.flatten() * multiplier.flatten(), #use repl_ordered calculated above
            'repl_ord': repl_ordered.flatten(),
            'action': action.flatten(),
            'multiplier': multiplier.flatten(),
            'mapped_dem': mapped_demand_pr.flatten(),
            'repl_rec': repl_received.flatten(),
            'mapped_forecast': self.mapped_forecast_plotting,
            'uot_plan': uot_plan,
            'adjusted_inventory_sla': x,
            'reward':reward
        }

        return self.state, reward, done, info


    def calculate_reward(self, inventory, demand, mapped_demand):
        # Ensure all inputs are tensors on the correct device
        # mapped_demand = torch.mm(demand.T, pl_ratio)[:, :self.num_fcs]
        demand = torch.from_numpy(demand)
        mapped_demand = torch.from_numpy(mapped_demand)
        sales = torch.min(inventory, mapped_demand)
        holding_cost = self.hc * torch.clamp(inventory - mapped_demand, min=0)
        shortage_cost = self.sc * torch.clamp(mapped_demand - inventory, min=0)
        sales_revenue = self.sc * sales * 5

        reward = sales_revenue - holding_cost - shortage_cost

        # Vector Reward
        reward = torch.sum(reward, dim=0)  # Shape: (num_fcs,)

        # For Scalar Reward, uncomment the following line:
        reward = torch.sum(reward)

        return reward, sales, holding_cost, shortage_cost, mapped_demand

    def calculate_reward_OT(self, inventory, realized_demand, mapped_dem_pr, benefits_data_table, compute_sla):
        # Move all inputs to the correct device
        inventory = inventory.reshape(-1, 1)
        realized_demand = torch.from_numpy(realized_demand)
        mapped_dem_pr = torch.from_numpy(mapped_dem_pr)
        # benefits_data_table = torch.tensor(benefits_data_table)


        temp_realized_demand = torch.cat([realized_demand, torch.zeros((1, self.num_sku))],dim=0).int()
        temp_inventory = torch.cat([inventory, torch.zeros((1, self.num_sku))], dim=0).int()

        demand_sums = torch.sum(temp_realized_demand[:-1, :], dim=0)
        inventory_sums = torch.sum(temp_inventory[:-1, :], dim=0)

        differences = demand_sums - inventory_sums

        # Adjust dummy rows
        temp_inventory[-1, :] = torch.clamp(differences, min=0)
        temp_realized_demand[-1, :] = torch.clamp(-differences, min=0)

        reward = 0
        holding_cost_ot, shortage_cost_ot = 0, 0
        holding_cost_pr, shortage_cost_pr = 0, 0

        for i in range(temp_inventory.shape[1]):
            ### M will be used for cost matrix in OT as well as for optimisation function for calculating SLA

            benefits_data_table_array = np.array(benefits_data_table)
            min_val, max_val = np.min(benefits_data_table_array), np.max(benefits_data_table_array)
            M = 1.0 - ((benefits_data_table_array - min_val) / (max_val - min_val)) #Full Norm
            #M = 1 - self.min_max_scaler.fit_transform(benefits_data_table_array)   #Column Norm

            #### For SLA Computation
            ### Use this only during testing:
            x = np.zeros((self.num_fcs))
            if compute_sla:
                M1 = 1.0 - ((benefits_data_table_array - min_val) / (max_val - min_val))
                alpha, beta = 0.1, 1000
                num_regions = M1.shape[1]
                tp = cp.Variable((self.num_fcs, num_regions), nonneg=True)  #Transport plan
                x = cp.Variable(self.num_fcs)                               #Inventory adjustment variable
                #Objective
                objective = cp.Minimize(alpha * cp.sum(cp.maximum(0,x)) + beta * cp.sum(cp.multiply(tp,M1)))
                #Constraints
                t1 = realized_demand.to('cpu').numpy().reshape(1,-1)
                t2 = inventory.to('cpu').numpy().reshape(1,-1)
                constraints = [cp.sum(tp,axis=0)==t1, cp.sum(tp,axis=1)==t2+x]
                #problem
                prob = cp.Problem(objective, constraints)
                prob.solve()
                optimization_fulfilment_cost = np.sum(np.multiply(tp.value,M1))
                x = x.value

            benefits = 1 - M
            M = torch.tensor(M)
            M = torch.cat([M, torch.ones((1, M.shape[1])) * self.sc], dim=0)
            M = torch.cat([M, torch.ones((M.shape[0], 1)) * self.hc], dim=1)

            supply = temp_inventory[:, i].cpu().numpy()
            demand = temp_realized_demand[:, i].cpu().numpy()
            
            # scaling costs for uot
            uot_scale_factor = M.max()
            uot = ot.solve(M.cpu().numpy() / uot_scale_factor.item(), supply.astype('float'), demand.astype('float'),method='emd')

            sales_at_FC = torch.tensor(uot.plan[:-1, :-1].sum(axis=1))
            mapped_dem = sales_at_FC.int()

            # (OT) Calculate reward
            holding_cost_ot += torch.clamp(temp_inventory[:-1, i].sum(), min=0) * self.hc * uot_scale_factor
            holding_qty_ot = torch.clamp(-differences[i], min=0)
            shortage_cost_ot += torch.clamp(differences[i], min=0) * self.sc * uot_scale_factor
            shortage_qty_ot = torch.clamp(differences[i], min=0)

            # (PR) Calculate reward
            diff = mapped_dem_pr.flatten() - inventory.flatten()
            holding_cost_pr = torch.where(diff > 0, torch.tensor(0.), diff * -self.hc)
            holding_qty_pr = torch.where(diff > 0, torch.tensor(0.), -diff)
            # holding_qty_pr = torch.sum(torch.where(diff > 0, torch.tensor(0.), -diff))

            shortage_cost_pr = torch.where(diff > 0, diff * self.sc, torch.tensor(0.))
            shortage_qty_pr = torch.where(diff > 0, diff, torch.tensor(0.))
            # shortage_qty_pr = torch.sum(torch.where(diff > 0, diff, torch.tensor(0.)))

            price = self.sc * 5
            sales_revenue = price * sales_at_FC

            local_cost = torch.sum(shortage_cost_pr + holding_cost_pr)
            global_cost = uot.value * uot_scale_factor.item()

            reward = -1 * (0 * global_cost + 1 * local_cost) 

            #All scaling for holding & shoratge is removed
            reward = reward / uot_scale_factor.item() #--> Rescaled to be in sync with older reward - this is temporary & needs to be removed

        return reward, sales_at_FC, holding_cost_pr, shortage_cost_pr, mapped_dem, holding_qty_pr, shortage_qty_pr, holding_qty_ot, shortage_qty_ot, uot.plan, x