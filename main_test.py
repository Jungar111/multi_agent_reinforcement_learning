"""Main file for project."""
from __future__ import print_function

from datetime import datetime
import typing as T
import numpy as np

from tqdm import trange

import wandb
import copy
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.algos.sac import SAC
from multi_agent_reinforcement_learning.data_models.config import A2CConfig, SACConfig
from multi_agent_reinforcement_learning.data_models.logs import ModelLog
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario

# from multi_agent_reinforcement_learning.utils.argument_parser import args_to_config
from multi_agent_reinforcement_learning.utils.sac_argument_parser import (
    args_to_config as SAC_args_to_config,
)
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.setup_grid import setup_dummy_grid
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    ActorEvaluator,
)

logger = init_logger()


def _train_loop(
    n_episodes: int,
    actor_data: T.List[ActorData],
    env: AMoD,
    models: T.List[ActorCritic],
    n_actions: int,
    episode_length: int,
    training: bool = True,
):
    """General train loop.

    Used both for testing and training, by setting training.
    """
    best_reward = -np.inf
    epochs = trange(n_episodes)
    for i_episode in epochs:
        for model in models:
            model.actor_data.model_log = ModelLog()
        env.reset()  # initialize environment

        all_actions = np.zeros(
            (
                len(models),
                episode_length,
                np.max(list(models[0].actor_data.flow.pax_flow.keys())) + 1,
            )
        )
        o = [None, None]
        actions = None

        for step in range(episode_length):
            if step > 0:
                obs_old = copy.deepcopy(o)
            # take matching step (Step 1 in paper)
            actor_data, done = env.pax_step(
                cplex_path=config.cplex_path,
                path=config.path,
            )
            for actor in models:
                actor.actor_data.model_log.reward += actor.actor_data.rewards.pax_reward
            # use GNN-RL policy (Step 2 in paper)
            for idx, model in enumerate(models):
                o[idx] = model.obs_parser.parse_obs(obs=actor_data[idx].graph_state)

            if step > 0:
                # store transition in memroy
                for idx, model in enumerate(models):
                    pax_and_reb_reward = (
                        model.actor_data.rewards.pax_reward
                        + model.actor_data.rewards.reb_reward
                    )
                    model.replay_buffer.store(
                        obs_old[idx],
                        actions[idx],
                        config.rew_scale * pax_and_reb_reward,
                        o[idx],
                    )

            actions = []
            for model in models:
                model.train_log.reward += model.actor_data.rewards.pax_reward
                actions.append(
                    model.select_action(
                        model.actor_data.graph_state, probabilistic=training
                    )
                )

            for idx, action in enumerate(actions):
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                models[idx].actor_data.flow.desired_acc = {
                    env.region[i]: int(
                        action[i]
                        * dictsum(actor_data[idx].graph_state.acc, env.time + 1)
                    )
                    for i in range(n_actions)
                }

                all_actions[idx, step, :] = list(
                    models[idx].actor_data.flow.desired_acc.values()
                )

            # solve minimum rebalancing distance problem (Step 3 in paper)

            solveRebFlow(env, config.path, config.cplex_path)

            actor_data, done = env.reb_step()

            for model in models:
                model.train_log.reward += model.actor_data.rewards.reb_reward
            # track performance over episode
            for model in models:
                model.actor_data.model_log.reward += model.actor_data.rewards.reb_reward
                model.actor_data.model_log.served_demand += (
                    model.actor_data.info.served_demand
                )
                model.actor_data.model_log.rebalancing_cost += (
                    model.actor_data.info.rebalancing_cost
                )
                model.rewards.append(
                    model.actor_data.rewards.pax_reward
                    + model.actor_data.rewards.reb_reward
                )

                model.train_log.served_demand += model.actor_data.info.served_demand
                model.train_log.rebalancing_cost += (
                    model.actor_data.info.rebalancing_cost
                )
            # stop episode if terminating conditions are met
            if done:
                break
            # Training loop
            if i_episode > 10:
                # sample from memory and update model
                for model in models:
                    batch = model.replay_buffer.sample_batch(
                        config.batch_size, norm=False
                    )
                    model.update(data=batch)

        # TODO What to do about training?
        # if training:
        #     # perform on-policy backprop
        #     for model in models:
        #         model.training_step()

        # Send current statistics to screen
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {actor_data[0].model_log.reward:.2f} |"
            f"ServedDemand: {actor_data[0].model_log.served_demand:.2f} "
            f"| Reb. Cost: {actor_data[0].model_log.rebalancing_cost:.2f}"
        )

        # Checkpoint best performing model
        if training:
            if (
                sum([model.actor_data.model_log.reward for model in models])
                > best_reward
            ):
                for model in models:
                    model.save_checkpoint(
                        path=f"./{config.directory}/ckpt/{config.path}/a2c_gnn_{model.actor_data.name}.pth"
                    )
                    best_reward = sum(
                        [model.actor_data.model_log.reward for model in models]
                    )
                    wandb.log({"Best Reward": best_reward})
        # Log KPIs on weights and biases

        for model in models:
            wandb.log({**model.actor_data.model_log.dict(model.actor_data.name)})

        if not training:
            return all_actions


def main(config: SACConfig):
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
        if isinstance(config, A2CConfig):
            scenario = Scenario(
                config=config,
                json_file=str(config.json_file),
                sd=config.seed,
                demand_ratio=config.demand_ratio,
                json_hr=config.json_hr,
                json_tstep=config.json_tstep,
                actor_data=actor_data,
            )
        elif isinstance(config, SACConfig):
            scenario = Scenario(
                config=config,
                json_file=str(config.json_file),
                sd=config.seed,
                demand_ratio=config.demand_ratio[config.city],
                json_hr=config.json_hr[config.city],
                json_tstep=config.json_tstep,
                actor_data=actor_data,
            )
        else:
            raise ValueError("Asger is über gay, ps. config error.")

    if isinstance(config, A2CConfig):
        env = AMoD(
            scenario=scenario, beta=config.beta, actor_data=actor_data, config=config
        )
        # Initialize A2C-GNN
        rl1_actor = ActorCritic(
            env=env, input_size=21, config=config, actor_data=actor_data[0]
        )
        rl2_actor = ActorCritic(
            env=env, input_size=21, config=config, actor_data=actor_data[1]
        )
    elif isinstance(config, SACConfig):
        env = AMoD(
            beta=config.beta[config.city],
            scenario=scenario,
            config=config,
            actor_data=actor_data,
        )
        # Initialise SAC
        rl1_actor = SAC(
            env=env,
            config=config,
            actor_data=actor_data[0],
            input_size=13,
            hidden_size=config.hidden_size,
            p_lr=config.p_lr,
            q_lr=config.q_lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            use_automatic_entropy_tuning=False,
            clip=config.clip,
            critic_version=config.critic_version,
        )
        rl2_actor = SAC(
            env=env,
            config=config,
            actor_data=actor_data[1],
            input_size=13,
            hidden_size=config.hidden_size,
            p_lr=config.p_lr,
            q_lr=config.q_lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            use_automatic_entropy_tuning=False,
            clip=config.clip,
            critic_version=config.critic_version,
        )
    else:
        raise ValueError("Asger er gay. PS. Config fejl igen.")

    models = [rl1_actor, rl2_actor]
    episode_length = config.max_steps  # set episode length
    n_actions = len(env.region)

    if not config.test:
        train_episodes = config.max_episodes  # set max number of training episodes
        for model in models:
            model.train()

        _train_loop(
            train_episodes,
            actor_data,
            env,
            models,
            n_actions,
            episode_length,
            training=True,
        )

    else:
        # Load pre-trained model
        rl1_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/{config.path}/a2c_gnn_{rl1_actor.actor_data.name}.pth"
        )
        rl2_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/{config.path}/a2c_gnn_{rl2_actor.actor_data.name}.pth"
        )

        test_episodes = 1
        episode_length = config.max_steps

        all_actions = _train_loop(
            test_episodes,
            actor_data,
            env,
            models,
            n_actions,
            episode_length,
            training=False,
        )

        actor_evaluator = ActorEvaluator()
        actor_evaluator.plot_average_distribution(
            actions=np.array(all_actions),
            T=episode_length,
            models=models,
        )

        # actor_evaluator.plot_distribution_at_time_step_t(
        #     actions=np.array(all_actions), models=models
        # )

    wandb.finish()


if __name__ == "__main__":
    config = SAC_args_to_config()
    config.wandb_mode = "disabled"
    # config.test = True
    # config.max_episodes = 10000
    # config.json_file = None
    # config.grid_size_x = 2
    # config.grid_size_y = 3
    # config.tf = 20
    main(config)