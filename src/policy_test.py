import os
import pickle
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import gym
from tqdm import tqdm
import rootutils
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.experimental import initialize, compose

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def extract_resolved_model_config():
    # Initialize Hydra with the correct config path
    with initialize(config_path="../configs"):
        cfg = compose(config_name="train")
        # Extract and resolve only the model configuration
        model_cfg = OmegaConf.select(cfg, "model")
        # model_cfg = OmegaConf.create(cfg.model)
        # resolved_model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

    return model_cfg

def add_item_features(state: torch.Tensor, item_ids, item_id: int) -> torch.Tensor:
    """
    Add item-specific features to state before passing to network
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Option 1: Concatenate one-hot encoded item_id
    item_idx = item_ids.index(item_id)
    one_hot = torch.zeros(len(item_ids), device=device)
    one_hot[item_idx] = 1

    # If state is not already a tensor, convert it
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, device=device)

    # Ensure state is on the correct device
    state = state.to(device)

    # Concatenate state and one-hot encoding
    return torch.cat([state, one_hot])

def multi_item_test_multi_episode(model_cfg, checkpoint_path, num_episodes=1000, test_mode=False, base_seed=42,new_item_ids=None):
    """
    Testing the policy from the saved checkpoint with multiple episodes
    """
    num_fcs = model_cfg["num_fcs"]
    total_steps = 63+84

    # Load model once
    model = hydra.utils.instantiate(model_cfg)
    print("Model Loaded")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    net = model.net
    net.eval()
    print("Uploaded checkpoint to model")


    # Prepare environment configuration
    cfg = OmegaConf.to_container(model_cfg.env, resolve=True)
    cfg["env_cfg"]["forecast_horizon"] = 75+84
    print("Item ids in config is: ", cfg["env_cfg"]["item_ids"])
    old_item_ids = cfg["env_cfg"]["item_ids"]
    if new_item_ids is not None:
        # Change the data pointing path to the different item_ids on which you want to test the policy
        cfg["env_cfg"]["item_ids"] = new_item_ids
        #cfg["env_cfg"]["data_dir"] = str(Path(cfg['env_cfg']['data_dir']).parent/ f"item_id_{new_item_nbr}")
        print(f"Running policy trained for {old_item_ids} on: {new_item_ids}")
    else:
        new_item_ids = cfg["env_cfg"]["item_ids"]

    print("Running Test Mode as ", test_mode)
    if test_mode:
        options = {'actual_test': True}
        cfg["env_cfg"]["starting_week"] = 12351 #12411
        cfg["env_cfg"]["test_week"] = 12351
    else:
        options = {'simulated_test': (0, 0)}
        cfg["env_cfg"]["starting_week"] = 12351 #12411
        cfg["env_cfg"]["test_week"] = 12351

    policy_results = dict()
    for item_id in new_item_ids:
        item_env_config = cfg.copy()
        item_env_config["env_cfg"]['item_nbr'] = item_id
        item_env_config["env_cfg"]["data_dir"] = str(Path(item_env_config['env_cfg']['data_dir']).parent/ f"item_id_{item_id}")

        # Initialize aggregation containers
        aggregate_metrics = defaultdict(lambda: np.zeros((num_episodes, total_steps)))
        aggregate_fc_metrics = defaultdict(lambda: np.zeros((num_episodes, total_steps)))
        aggregate_metrics_DoS = defaultdict(lambda: np.zeros((num_episodes, total_steps)))

        # Add these with other initializations
        sla_demand_metrics = np.zeros((num_episodes, total_steps, num_fcs + 1))  # +1 for overall
        sla_optimization_metrics = np.zeros((num_episodes, total_steps, num_fcs + 1))

        dos_metrics = {
            'actual': np.zeros((num_episodes, total_steps, num_fcs + 1)),  # actual excess
            'order': np.zeros((num_episodes, total_steps, num_fcs + 1)),  # order excess
            'projected': np.zeros((num_episodes, total_steps, num_fcs + 1))  # projected excess
        }

        # Store one detailed episode for visualization
        detailed_episode = None
        detailed_fc_data = None

        # Initialize cumulative Q-values array after first episode to get correct shape
        cumulative_q_values = None

        env = gym.make(item_env_config.get('id'), env_cfg=item_env_config["env_cfg"])
        # Run multiple episodes
        for episode in tqdm(range(num_episodes), desc=f"Running episodes for item:{item_id}"):
            # Set different seed for each episode
            episode_seed = base_seed + episode
            torch.manual_seed(episode_seed)
            np.random.seed(episode_seed)
            base_state = env.reset(seed=episode_seed, options=options)

            fc_st_dim = env.inventory_state.get_state_dim() // num_fcs
            # inventory_at_beginning_of_day = state[0::fc_st_dim].reshape(-1,num_fcs)
            # print(inventory_at_beginning_of_day)

            logs = defaultdict(list)
            episode_q_values = []  # Store Q-values for this episode

            for step in range(total_steps):
                # Add item features to the base state
                state = add_item_features(base_state, old_item_ids, item_id)

                state_tensor = state.clone().detach().float().unsqueeze(0)
                q_values = net(state_tensor)
                episode_q_values.append(q_values.detach().cpu().numpy())
                action = q_values.argmax(dim=-1).squeeze().cpu()

                action1 = 0 + action * 0.1
                next_state, reward, done, info = env.step(action1)

                dem_cp_fc = state_tensor[0][1::fc_st_dim]
                proj_oh_lt_fc = state_tensor[0][8::fc_st_dim]
                exp_inv_repl_FC = state_tensor[0][9::fc_st_dim]

                # Store metrics for this episode
                aggregate_metrics['total_inv_begin_day'][episode, step] = sum(info['inv_at_beginning_of_day'])
                aggregate_metrics['total_repl_rec'][episode, step] = sum(info['repl_rec'])
                aggregate_metrics['total_inv_after_replen'][episode, step] = sum(info['inv_after_replen'])
                aggregate_metrics['total_inv_at_end_of_day'][episode, step] = sum(info['inv_at_end_of_day'])
                aggregate_metrics['total_mapped_dem'][episode, step] = sum(info['mapped_dem'])
                aggregate_metrics['total_order_qty'][episode, step] = sum(info['repl_ord'])
                aggregate_metrics['total_shortage_ot_qty'][episode, step] = sum(info['shortage_qty_ot'])
                aggregate_metrics['total_holding_ot_qty'][episode, step] = sum(info['holding_qty_ot'])
                aggregate_metrics['total_mapped_forecast'][episode, step] = sum(info['mapped_forecast'])
                aggregate_metrics['total_prod_sales'][episode, step] = sum(info['sales_at_FC'])
                aggregate_metrics['total_dem_cp'][episode, step] = sum(dem_cp_fc).item()
                aggregate_metrics['total_proj_oh_lt'][episode, step] = sum(proj_oh_lt_fc).item()
                aggregate_metrics['total_exp_inv_repl'][episode, step] = sum(exp_inv_repl_FC).item()

                # SLA tracking for demand
                for i in range(num_fcs):
                    sla_demand_metrics[episode, step, i] = int(info['inv_after_replen'][i] >= info['mapped_dem'][i])
                    sla_optimization_metrics[episode, step, i] = int(info['adjusted_inventory_sla'][i] <= 0)
                    # Track DOS metrics
                    #dos_metrics['actual'][episode, step, i] = max(0, info['inv_after_replen'][i] - info['mapped_forecast'][i])
                    #Karthik - Changing actual excess to use mapped demand instead of mapped forecast
                    dos_metrics['actual'][episode, step, i] = max(0,info['inv_after_replen'][i] - info['mapped_dem'][i])
                    dos_metrics['order'][episode, step, i] = max(0, info['repl_ord'][i] - exp_inv_repl_FC[i])
                    dos_metrics['projected'][episode, step, i] = max(0,info['repl_ord'][i] + proj_oh_lt_fc[i] - dem_cp_fc[i])

                # Overall SLA
                sla_demand_metrics[episode, step, -1] = int(sum(info['inv_after_replen']) >= sum(info['mapped_dem']))
                sla_optimization_metrics[episode, step, -1] = int(sum(info['inv_after_replen']) >= sum(info['mapped_dem']))

                # Overall metrics (sum across FCs)
                dos_metrics['actual'][episode, step, -1] = sum(dos_metrics['actual'][episode, step, :-1])
                dos_metrics['order'][episode, step, -1] = sum(dos_metrics['order'][episode, step, :-1])
                dos_metrics['projected'][episode, step, -1] = sum(dos_metrics['projected'][episode, step, :-1])



                # Store FC-specific metrics
                for i in range(num_fcs):
                    aggregate_fc_metrics[f'inv_begin_FC{i}'][episode, step] = info['inv_at_beginning_of_day'][i]
                    aggregate_fc_metrics[f'repl_rec_FC{i}'][episode, step] = info['repl_rec'][i]
                    aggregate_fc_metrics[f'inv_repl_FC{i}'][episode, step] = info['inv_after_replen'][i]
                    aggregate_fc_metrics[f'inv_eod_FC{i}'][episode, step] = info['inv_at_end_of_day'][i]
                    aggregate_fc_metrics[f'mapped_dem_FC{i}'][episode, step] = info['mapped_dem'][i]
                    aggregate_fc_metrics[f'order_qty_FC{i}'][episode, step] = info['repl_ord'][i]
                    aggregate_fc_metrics[f'shrt_qty_pr_FC{i}'][episode, step] = info['shortage_qty_pr'][i]
                    aggregate_fc_metrics[f'hldg_qty_pr_FC{i}'][episode, step] = info['holding_qty_pr'][i]
                    aggregate_fc_metrics[f'mapped_forecast_FC{i}'][episode, step] = info['mapped_forecast'][i]
                    aggregate_fc_metrics[f'sales_at_FC{i}'][episode, step] = info['sales_at_FC'][i]
                    aggregate_fc_metrics[f'dem_cp_FC{i}'][episode, step] = dem_cp_fc[i]
                    aggregate_fc_metrics[f'proj_oh_lt_FC{i}'][episode, step] = proj_oh_lt_fc[i]
                    aggregate_fc_metrics[f'exp_inv_repl_FC{i}'][episode, step] = exp_inv_repl_FC[i]
                    aggregate_fc_metrics[f'action_FC{i}'][episode, step] = action1[i]
                    aggregate_fc_metrics[f'multiplier_FC{i}'][episode, step] = info['multiplier'][i]

                # For collecting detailed data from first episode
                if episode == 0:
                    # Log data
                    logs['step'].append(step)
                    logs['total_inv_begin_day'].append(sum(info['inv_at_beginning_of_day']).item())
                    logs['total_repl_rec'].append(sum(info['repl_rec']).item())
                    logs['total_inv_after_replen'].append(sum(info['inv_after_replen']).item())
                    logs['total_inv_at_end_of_day'].append(sum(info['inv_at_end_of_day']).item())
                    logs['total_mapped_dem'].append(sum(info['mapped_dem']))
                    logs['total_order_qty'].append(sum(info['repl_ord']).item())
                    logs['total_shortage_ot_qty'].append(sum(info['shortage_qty_ot']).item())
                    logs['total_holding_ot_qty'].append(sum(info['holding_qty_ot']).item())
                    logs['total_mapped_forecast'].append(sum(info['mapped_forecast']))
                    logs['total_prod_sales'].append(sum(info['sales_at_FC']).item())
                    logs['total_dem_cp'] = sum(dem_cp_fc).item()
                    logs['total_proj_oh_lt'] = sum(proj_oh_lt_fc).item()
                    logs['total_exp_inv_repl'] = sum(exp_inv_repl_FC).item()

                    for i in range(num_fcs):
                        logs[f'inv_begin_FC{i}'].append(info['inv_at_beginning_of_day'][i].item())
                        logs[f'repl_rec_FC{i}'].append(info['repl_rec'][i].item())
                        logs[f'inv_repl_FC{i}'].append(info['inv_after_replen'][i].item())
                        logs[f'inv_eod_FC{i}'].append(info['inv_at_end_of_day'][i].item())
                        logs[f'mapped_dem_FC{i}'].append(info['mapped_dem'][i].item())
                        logs[f'order_qty_FC{i}'].append(info['repl_ord'][i].item())
                        logs[f'shrt_qty_pr_FC{i}'].append(info['shortage_qty_pr'][i].item())
                        logs[f'hldg_qty_pr_FC{i}'].append(info['holding_qty_pr'][i].item())
                        logs[f'mapped_forecast_FC{i}'].append(info['mapped_forecast'][i].item())
                        logs[f'sales_at_FC{i}'].append(info['sales_at_FC'][i].item())
                        logs[f'dem_cp_FC{i}'].append(dem_cp_fc[i].item())
                        logs[f'proj_oh_lt_FC{i}'].append(proj_oh_lt_fc[i].item())
                        logs[f'exp_inv_repl_FC{i}'].append(exp_inv_repl_FC[i].item())
                        logs[f'action_FC{i}'].append(action1[i].item())
                        logs[f'multiplier_FC{i}'].append(info['multiplier'][i].item())

                base_state = next_state
                if done:
                    break


            # Process Q-values for this episode
            episode_q_values_array = np.array(episode_q_values)
            if cumulative_q_values is None:
                cumulative_q_values = episode_q_values_array
            else:
                cumulative_q_values += episode_q_values_array

            # Store detailed data from first episode
            if episode == 0:
                detailed_episode = create_overall_df(logs)
                detailed_fc_data = create_fc_df(logs, num_fcs, env, total_steps)

            env.close()

        # Calculate average Q-values across all episodes
        average_q_values = cumulative_q_values / num_episodes
        average_q_values_df = create_q_values_df(average_q_values, total_steps)

        # Calculate aggregate statistics
        aggregate_stats = calculate_aggregate_statistics(aggregate_metrics, aggregate_fc_metrics, num_fcs, total_steps)

        # Before return statement
        sla_demand_avg = np.mean(sla_demand_metrics, axis=0)  # Average across episodes
        sla_optimization_avg = np.mean(sla_optimization_metrics, axis=0)  # Average across episodes

        # Average across episodes
        dos_avg = {metric: np.mean(values, axis=0) for metric, values in dos_metrics.items()}


        sla_df = pd.DataFrame({
            'Step': range(total_steps),
            'Overall_Demand_SLA': sla_demand_avg[:, -1],
            'Overall_Optimization_SLA': sla_optimization_avg[:, -1],
            **{f'FC{i}_Demand_SLA': sla_demand_avg[:, i] for i in range(num_fcs)},
            **{f'FC{i}_Optimization_SLA': sla_optimization_avg[:, i] for i in range(num_fcs)}
        })

        dos_df = pd.DataFrame({
            'Step': range(total_steps),
            'Overall_Actual_Excess': dos_avg['actual'][:, -1],
            'Overall_Order_Excess': dos_avg['order'][:, -1],
            'Overall_Projected_Excess': dos_avg['projected'][:, -1],
            **{f'FC{i}_Actual_Excess': dos_avg['actual'][:, i] for i in range(num_fcs)},
            **{f'FC{i}_Order_Excess': dos_avg['order'][:, i] for i in range(num_fcs)},
            **{f'FC{i}_Projected_Excess': dos_avg['projected'][:, i] for i in range(num_fcs)}
        })
        policy_results[item_id] = [detailed_episode, detailed_fc_data, average_q_values_df, aggregate_stats, sla_df, dos_df]
    return policy_results


def calculate_aggregate_statistics(aggregate_metrics, aggregate_fc_metrics, num_fcs, total_steps):
    """Calculate mean, std, min, max for all metrics across episodes"""
    stats = {}

    # Overall metrics
    for metric_name, data in aggregate_metrics.items():
        stats[f'{metric_name}_mean'] = np.mean(data, axis=0)
        stats[f'{metric_name}_std'] = np.std(data, axis=0)
        stats[f'{metric_name}_min'] = np.min(data, axis=0)
        stats[f'{metric_name}_max'] = np.max(data, axis=0)

    # FC-specific metrics
    for metric_name, data in aggregate_fc_metrics.items():
        stats[f'{metric_name}_mean'] = np.mean(data, axis=0)
        stats[f'{metric_name}_std'] = np.std(data, axis=0)
        stats[f'{metric_name}_min'] = np.min(data, axis=0)
        stats[f'{metric_name}_max'] = np.max(data, axis=0)

    # Convert to DataFrames
    aggregate_dfs = {}

    # Overall metrics DataFrame
    steps = np.arange(total_steps)
    for metric_type in ['mean', 'std', 'min', 'max']:
        df_data = {
            'Step': steps,
            **{metric: stats[f'{metric}_{metric_type}'] for metric in aggregate_metrics.keys()}
        }
        aggregate_dfs[f'overall_{metric_type}'] = pd.DataFrame(df_data)

    # FC metrics DataFrame
    for metric_type in ['mean', 'std', 'min', 'max']:
        df_data = {
            'Step': steps,
            **{metric: stats[f'{metric}_{metric_type}'] for metric in aggregate_fc_metrics.keys()}
        }
        aggregate_dfs[f'fc_{metric_type}'] = pd.DataFrame(df_data)

    return aggregate_dfs


def create_overall_df(logs):
    """Create overall metrics DataFrame"""
    return pd.DataFrame({
        'Step': logs['step'],
        'Total_Inv_Begin': logs['total_inv_begin_day'],
        'Total_Repl_Rec': logs['total_repl_rec'],
        'Total_Inv_Repl': logs['total_inv_after_replen'],
        'Total_Inv_End': logs['total_inv_at_end_of_day'],
        'Total_Mapped_Dem': logs['total_mapped_dem'],
        'Total_Order_Qty': logs['total_order_qty'],
        'Total_shortage_ot_qty': logs['total_shortage_ot_qty'],
        'Total_holding_ot_qty': logs['total_holding_ot_qty'],
        'Total_Mapped_Forecast': logs['total_mapped_forecast'],
        'Total_Simulated_Sales': logs['total_prod_sales'],
        'Total_Dem_CP': logs['total_dem_cp'],
        'Total_ProjectedOH_LT': logs['total_proj_oh_lt'],
        'Total_ExpInv_Repl': logs['total_exp_inv_repl']
    })


def create_fc_df(logs, num_fcs, env, total_steps):
    """Create FC-specific metrics DataFrame"""

    fc_columns = [f'{col}_FC{i}' for i in range(num_fcs) for col in
                  ['inv_begin','repl_rec','inv_repl','inv_eod', 'mapped_dem','order_qty', 'shrt_qty_pr', 'hldg_qty_pr',
                   'mapped_forecast', 'sales_at', 'dem_cp', 'proj_oh_lt','exp_inv_repl','action','multiplier']]

    fc_df = pd.DataFrame({col: logs[col] for col in fc_columns})
    lt_fc_df = pd.DataFrame(env.inventory_state.action_pipeline[:, 3:total_steps-3, 2].numpy().T,
                            columns=[f"LT_FC{i}" for i in range(num_fcs)])

    fc_df = pd.concat([pd.DataFrame(logs["step"], columns=["Step"]), fc_df, lt_fc_df], axis=1)

    return fc_df


def create_q_values_df(q_values_array, total_steps):
    """Create Q-values DataFrame"""
    _, _, i, j = q_values_array.shape
    columns = [f'FC_{x}_action_{y}' for x in range(i) for y in range(j)]
    return pd.DataFrame(q_values_array.reshape(total_steps, -1), columns=columns)


if __name__ == "__main__":
    root = rootutils.setup_root(__file__, pythonpath=True)
    print(root)
    #Specify the item number to be tested
    #item_ids = [873764,36717960,167121557,873685]
    item_ids = [873764,36717960,167121557]

    # Specify the model path to be tested
    #model_run_timestamp = "2025-01-06_13-15-00"
    model_run_timestamp = "2025-01-09_16-30-48"
    checkpoint_path = root / f"logs/train/runs/{model_run_timestamp}/checkpoints/final_model.ckpt"
    #Specify the config name to be tested & saved
    run_name_prefix = "3FC_multi-item_model_value-correction"
    test_mode_configs = {'-test_sim': [False,1000] , '-test_act': [True,2]}
    for j,v in test_mode_configs.items():
        test_mode = v[0]
        model_cfg = extract_resolved_model_config()

        # Run multi-episode test
        policy_results = multi_item_test_multi_episode(
            model_cfg,
            checkpoint_path,
            num_episodes=v[1],
            test_mode=test_mode,
            base_seed=42,
            new_item_ids = item_ids
        )
        for item_id in item_ids:
            [detailed_df, detailed_fc_df, q_values_df, aggregate_stats, sla_df, dos_df] = policy_results[item_id]
            run_name = run_name_prefix + f'_{item_id}'
            # Create save directory
            save_log_path = root / f"notebooks/policy_logs/{run_name}{j}"
            if not os.path.exists(save_log_path):
                os.makedirs(save_log_path)
                print(f"Folder '{save_log_path}' created.")
            else:
                print(f"Folder '{save_log_path}' already exists.")

            # Save detailed results (from first episode)
            detailed_df.to_csv(save_log_path / "test_policy_overall_config.csv", index=False)
            detailed_fc_df.to_csv(save_log_path / "test_policy_FC_config.csv", index=False)
            q_values_df.to_csv(save_log_path / "q_values_analysis.csv", index=False)
            sla_df.to_csv(save_log_path / "sla_analysis.csv", index=False)
            dos_df.to_csv(save_log_path / "dos_analysis.csv", index=False)

            # Save aggregate statistics
            for metric_type in ['mean', 'std', 'min', 'max']:
                aggregate_stats[f'overall_{metric_type}'].to_csv(
                    save_log_path / f"aggregate_overall_{metric_type}.csv", index=False)
                aggregate_stats[f'fc_{metric_type}'].to_csv(
                    save_log_path / f"aggregate_fc_{metric_type}.csv", index=False)