"""Main file for project."""
from __future__ import print_function

from datetime import datetime

from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.data_models.config import Config
from multi_agent_reinforcement_learning.data_models.logs import ModelLog
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.utils.argument_parser import args_to_config
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.setup_grid import setup_dummy_grid
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    ActorEvaluator,
)

logger = init_logger()


def _train_loop(epochs, actor_data, env, models, n_actions, T, backprop=True):
    for i_episode in epochs:
        for model in models:
            model.actor_data.model_log = ModelLog()
        env.reset()  # initialize environment
        for step in range(T):
            # take matching step (Step 1 in paper)
            actor_data, done, ext_done = env.pax_step(
                cplex_path=config.cplex_path, path="scenario_nyc4"
            )
            for model in models:
                model.actor_data.model_log.reward += model.actor_data.pax_reward
            # use GNN-RL policy (Step 2 in paper)

            actions = []
            for model in models:
                model.train_log.reward += model.actor_data.pax_reward
                actions.append(model.select_action(model.actor_data.obs))

            for idx, action in enumerate(actions):
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                models[idx].actor_data.desired_acc = {
                    env.region[i]: int(
                        action[i] * dictsum(actor_data[idx].acc, env.time + 1)
                    )
                    for i in range(n_actions)
                }

            # solve minimum rebalancing distance problem (Step 3 in paper)

            solveRebFlow(env, "scenario_nyc4", config.cplex_path)

            actor_data, done = env.reb_step()

            for model in models:
                model.train_log.reward += model.actor_data.reb_reward
            # track performance over episode
            for model in models:
                model.actor_data.model_log.reward += model.actor_data.reb_reward
                model.actor_data.model_log.served_demand += (
                    model.actor_data.info.served_demand
                )
                model.actor_data.model_log.rebalancing_cost += (
                    model.actor_data.info.rebalancing_cost
                )
                model.rewards.append(
                    model.actor_data.pax_reward + model.actor_data.reb_reward
                )

                model.train_log.served_demand += model.actor_data.info.served_demand
                model.train_log.rebalancing_cost += (
                    model.actor_data.info.rebalancing_cost
                )
            # stop episode if terminating conditions are met
            if done:
                break

        if backprop:
            # perform on-policy backprop
            for model in models:
                model.training_step()

        # Send current statistics to screen
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {actor_data[0].model_log.reward:.2f} |"
            f"ServedDemand: {actor_data[0].model_log.served_demand:.2f} "
            f"| Reb. Cost: {actor_data[0].model_log.rebalancing_cost:.2f}"
        )

        # Checkpoint best performing model
        for model in models:
            if model.actor_data.model_log.reward >= model.actor_data.best_reward:
                model.save_checkpoint(
                    path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_{model.actor_data.name}.pth"
                )
                model.actor_data.best_reward = model.actor_data.model_log.reward
                wandb.log(
                    {
                        f"Best Reward {model.actor_data.name}": model.actor_data.best_reward
                    }
                )
        # Log KPIs on weights and biases

        for model in models:
            wandb.log({**model.actor_data.model_log.dict(model.actor_data.name)})


def main(config: Config):
    """Run main training loop."""
    logger.info("Running main loop.")

    advesary_number_of_cars = int(config.total_number_of_cars / 2)

    actor_data = [
        ActorData(
            name="RL_1", no_cars=config.total_number_of_cars - advesary_number_of_cars
        ),
        ActorData(name="RL_2", no_cars=advesary_number_of_cars),
    ]

    wandb_config_log = {**vars(config)}
    for actor in actor_data:
        wandb_config_log[f"no_cars_{actor.name}"] = actor.no_cars

    wandb.init(
        mode=config.wandb_mode,
        project="master2023",
        name=f"test_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        if config.test
        else f"train_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        config=wandb_config_log,
    )

    # Define AMoD Simulator Environment
    if config.json_file is None:
        # Define variable for environment
        demand_ratio, demand_input = setup_dummy_grid(config, determ=True)
        scenario = Scenario(
            config=config,
            demand_ratio=demand_ratio,
            demand_input=demand_input,
            actor_data=actor_data,
        )
    else:
        scenario = Scenario(
            config=config,
            json_file=config.json_file,
            sd=config.seed,
            demand_ratio=config.demand_ratio,
            json_hr=config.json_hr,
            json_tstep=config.json_tsetp,
            actor_data=actor_data,
        )

    env = AMoD(scenario=scenario, beta=config.beta, actor_data=actor_data)
    # Initialize A2C-GNN
    rl1_actor = ActorCritic(
        env=env, input_size=21, config=config, actor_data=actor_data[0]
    )
    rl2_actor = ActorCritic(
        env=env, input_size=21, config=config, actor_data=actor_data[1]
    )

    models = [rl1_actor, rl2_actor]
    T = config.max_steps  # set episode length
    n_actions = len(env.region)

    if not config.test:
        #######################################
        #############Training Loop#############
        #######################################
        # Initialize lists for logging
        train_episodes = config.max_episodes  # set max number of training episodes
        epochs = trange(train_episodes)  # epoch iterator
        for model in models:
            model.train()

        _train_loop(epochs, actor_data, env, models, n_actions, T, backprop=True)

    else:
        # Load pre-trained model
        rl1_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
        )
        rl2_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
        )

        test_episodes = 1  # set max number of training episodes
        T = config.max_steps  # set episode length
        epochs = trange(test_episodes)  # epoch iterator
        # Initialize lists for logging

        _train_loop(epochs, actor_data, env, models, n_actions, T, backprop=False)

        actor_evaluator = ActorEvaluator(actor_data=actor_data)
        actor_evaluator.plot_average_distribution()

    wandb.finish()


if __name__ == "__main__":
    config = args_to_config()
    # config.wandb_mode = "disabled"
    config.max_episodes = 300
    config.json_file = None
    config.grid_size_x = 2
    config.grid_size_y = 3
    config.tf = 20
    main(config)
