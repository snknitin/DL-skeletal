from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning.pytorch.tuner import Tuner


rootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from src import gymenv   # This is important to register the environment
import warnings

# Suppress specific UserWarnings related to DataLoader
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have many workers.*")

### Customised Policy Testing
from policy_test2 import *
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.auto_tune:
        # Create a Tuner
        tuner = Tuner(trainer)
        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        tuner.lr_find(model,max_lr=0.9,min_lr=1e-7)
        # Auto-scale batch size with binary search
        #tuner.scale_batch_size(model, mode="binsearch")
        print("Tuned Learning Rate is :",model.hparams.lr)
        #print("Tuned Batch Size is :", model.hparams.batch_size)
        # model.env.reset()
        model.agent.reset()

    object_dict = {
        "cfg": cfg,
        # "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))
        items = list(cfg.item_ids)
        print(f"Items in the run : {items}")
        checkpoint_path = (f"{cfg.paths.output_dir}/checkpoints/final_model.ckpt")
        object_dict['checkpoint_path'] = checkpoint_path
        trainer.save_checkpoint(checkpoint_path)
        log.info(f"Final model saved to {checkpoint_path}")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        checkpoint_path = (f"{cfg.paths.output_dir}/checkpoints/final_model.ckpt")
        if os.path.exists(checkpoint_path):
            print("Checkpoint file found")
        else:
            print("Manual Checkpoint File does not exist.")
            checkpoint_path = trainer.checkpoint_callback.best_model_path
            if checkpoint_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                checkpoint_path = None


        # trainer.test(model=model, ckpt_path=checkpoint_path)
        log.info(f"Best ckpt path: {checkpoint_path}")


        # Run your policy test
        test_mode_configs = {'-test_sim': [False, 1000], '-test_act': [True, 2]}
        run_name_prefix = f"{cfg.get('run_name')}"

        for test_suffix, test_config in test_mode_configs.items():
            test_mode, num_episodes = test_config

            log.info(f"Running policy test with mode: {test_suffix}, episodes: {num_episodes}")

            # Run multi-episode test
            policy_results = multi_item_test_multi_episode(
                model_cfg=cfg.model,
                checkpoint_path=checkpoint_path,
                num_episodes=num_episodes,
                test_mode=test_mode,
                base_seed=cfg.seed,
                new_item_ids=cfg.get('item_ids')
            )

            # Save results for each item
            for item_id in cfg.get('item_ids'):
                [detailed_df, detailed_fc_df, q_values_df, aggregate_stats, sla_df, dos_df] = policy_results[item_id]

                num_items = len(cfg.get('item_ids'))
                run_name = run_name_prefix + f"{num_items}items_{cfg.get('num_fcs')}FCs_{item_id}"

                # Create save directory within the training output directory
                # save_log_path = Path(cfg.paths.root_dir) / f"notebooks/policy_logs/{run_name}{test_suffix}"

                save_log_path = Path(cfg.paths.output_dir) / f"{run_name}{test_suffix}"
                save_log_path.mkdir(parents=True, exist_ok=True)
                log.info(f"Saving test results to {save_log_path}")

                # Save all results
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

        log.info(f"Policy testing completed. Results saved in {cfg.paths.output_dir}/test_results/")



    print("Tuned Learning Rate was :", model.hparams.lr)
    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
