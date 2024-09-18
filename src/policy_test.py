import os
from pathlib import Path
import torch
import rootutils
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
#
# seed_ot = 42
# ot.utils.check_random_state(seed_ot)
#
# seed = 3407
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.experimental import initialize, compose


def extract_resolved_model_config():
    # Initialize Hydra with the correct config path
    with initialize(config_path="../configs"):
        cfg = compose(config_name="train")
        # Extract and resolve only the model configuration
        model_cfg = OmegaConf.select(cfg, "model")
        # model_cfg = OmegaConf.create(cfg.model)
        # resolved_model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

    return model_cfg


def dqn_test(model_cfg,checkpoint_path):
    """
    Testing the policy from the saved chekcpoint
    :param model_cfg:
    :param checkpoint_path:
    :return:
    """
    num_fcs = model_cfg["num_fcs"]

    # Change this as needed and test code first
    # start_week =  12351
    # model_cfg.env.env_cfg.starting_week = start_week

    # Instantiate the model using Hydr
    model = hydra.utils.instantiate(model_cfg)
    print("Model Loaded")
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load the state dict into the model
    model.load_state_dict(checkpoint['state_dict'])
    print("Uploaded checkpoint to model")

    # Get the weights of net and target_net
    net_weights = model.net.state_dict()
    target_net_weights = model.target_net.state_dict()

    env = model.env
    agent = model.agent
    net = model.net
    net.eval()

    print("Running Test Mode")
    options = {'test': True}
    state = env.reset(options=options)

    total_steps = 60 # test data size = 60 days
    logs = []
    logs_FC = []

    for i in range(total_steps):
        logs_curr = []
        logs_FC_curr = []
        # print("step",i)
        reward, done, info = agent.play_step(net, epsilon=0)
        logs_curr += [sum(info['inv_after_replen']), sum(info['mapped_dem']), sum(info['inv_at_end_of_day'])]

        column_name = []
        for i in range(num_fcs):
            logs_FC_curr += [info['inv_after_replen'][i].item(), info['shortage_cost'][i], info['holding_cost'][i],
                             info['mapped_dem'][i]]
            column_name.append(f'inv_repl_FC{i}')
            column_name.append(f'shrt_cost_FC{i}')
            column_name.append(f'hldg_cost_FC{i}')
            column_name.append(f'total_dem_FC{i}')

        logs.append(logs_curr)
        logs_FC.append(logs_FC_curr)

    column_name = []
    for i in range(num_fcs):
        column_name.append(f'inv_repl_FC{i}')
        column_name.append(f'shrt_qty_FC{i}')
        column_name.append(f'hldg_qty_FC{i}')
        column_name.append(f'total_dem_FC{i}')

    log_df = pd.DataFrame(np.round(logs, 2), columns=['Inv_Repl', 'Mapped_Dem', 'Inv_end'])
    fc_log_df = pd.DataFrame(logs_FC,columns=column_name)

    return log_df, fc_log_df

if __name__ == "__main__":
    root = rootutils.setup_root(__file__, pythonpath=True)
    print(root)

    run_name = "3FC_400k_SS_replicate"
    checkpoint_path = root / "src/correct_3fc_ss.ckpt"
    config_dir = root / "configs/"

    model_cfg = extract_resolved_model_config()
    # print(model_cfg)

    logs1, logs2 = dqn_test(model_cfg, checkpoint_path)
    save_log_path = root/f"notebooks/policy_logs/{run_name}"

    # Check if the folder exists
    if not os.path.exists(save_log_path):
        # Create the folder
        os.makedirs(save_log_path)
        print(f"Folder '{save_log_path}' created.")
    else:
        print(f"Folder '{save_log_path}' already exists.")

    # Saving files
    logs1.to_csv(save_log_path/"test_policy_overall_config.csv")
    logs2.to_csv(save_log_path/"test_policy_FC_config.csv")