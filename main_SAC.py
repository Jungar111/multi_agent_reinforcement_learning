"""Main file for the SAC implementation for the project."""
from __future__ import print_function
from tqdm import trange
import numpy as np
import wandb
from datetime import datetime
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.algos.sac import SAC
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.algos.sac_gnn_parser import GNNParser
from multi_agent_reinforcement_learning.data_models.logs import ModelLog
from multi_agent_reinforcement_learning.utils.sac_argument_parser import args_to_config
from multi_agent_reinforcement_learning.data_models.city_enum import City
from multi_agent_reinforcement_learning.data_models.config import SACConfig
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair
import copy

logger = init_logger()


def main(config: SACConfig):
    """Main loop for training and testing."""
    advesary_number_of_cars = int(config.total_number_of_cars / 2)
    actor_data = [
        ActorData(
            name="RL_1", no_cars=config.total_number_of_cars - advesary_number_of_cars
        ),
        ActorData(name="RL_2", no_cars=advesary_number_of_cars),
    ]

    wandb.init(
        mode=config.wandb_mode,
        project="master2023",
        name=f"train_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
    )

    logging_dict = {}
    if not config.test:
        """Run main training loop."""
        logger.info("Running main training loop for SAC.")
        scenario = Scenario(
            config=config,
            json_file=str(config.json_file),
            sd=config.seed,
            demand_ratio=config.demand_ratio[config.city],
            json_hr=config.json_hr[config.city],
            json_tstep=config.json_tstep,
            actor_data=actor_data,
        )
        env = AMoD(
            beta=config.beta[config.city],
            scenario=scenario,
            config=config,
            actor_data=actor_data,
        )
        # Timehorizon T=6 (K in paper)
        parser = GNNParser(env, T=6, json_file=f"data/scenario_{config.city}.json")
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
        model_data_pairs = [
            ModelDataPair(rl1_actor, actor_data[0]),
            ModelDataPair(rl2_actor, actor_data[1]),
        ]
        train_episodes = config.max_episodes
        # T = config.max_steps
        epochs = trange(train_episodes)
        best_reward = -np.inf
        # best_reward_test = -np.inf

        wandb_config_log = {**vars(config)}
        for model in model_data_pairs:
            wandb_config_log[f"test_{model.actor_data.name}"] = model.actor_data.no_cars

        for model_data_pair in model_data_pairs:
            model_data_pair.model.train()

        for i_episode in epochs:
            for model_data_pair in model_data_pairs:
                model_data_pair.actor_data.model_log = ModelLog()

            env.reset(model_data_pairs)  # initialize environment
            episode_reward = [0, 0]
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            done = False
            step = 0
            o = [None, None]
            action_rl = None
            obs_list = [None, None]
            while not done:
                # take matching step (Step 1 in paper)
                if step > 0:
                    obs_list[0] = copy.deepcopy(o[0])
                    obs_list[1] = copy.deepcopy(o[1])
                done = env.pax_step(
                    model_data_pairs=model_data_pairs,
                    cplex_path=config.cplex_path,
                    path=config.path,
                )
                for idx, model in enumerate(model_data_pairs):
                    o[idx] = parser.parse_obs(model.actor_data.graph_state)
                    episode_reward[idx] += model.actor_data.rewards.pax_reward

                for idx, model in enumerate(model_data_pairs):
                    if step > 0:
                        # store transition in memory
                        rl_reward = (
                            model.actor_data.rewards.pax_reward
                            + model.actor_data.rewards.reb_reward
                        )
                        model.model.replay_buffer.store(
                            obs_list[idx],
                            action_rl,
                            config.rew_scale * rl_reward,
                            o[idx],
                        )
                    action_rl = model.model.select_action(o[idx])
                for idx, model in enumerate(model_data_pairs):
                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    model.actor_data.flow.desired_acc = {
                        env.region[i]: int(
                            action_rl[i]
                            * dictsum(model.actor_data.graph_state.acc, env.time + 1)
                        )
                        for i in range(len(env.region))
                    }

                # solve minimum rebalancing distance problem (Step 3 in paper)
                solveRebFlow(
                    env,
                    config.path,
                    config.cplex_path,
                    model_data_pairs,
                )
                # Take action in environment
                done = env.reb_step(model_data_pairs)

                for idx, model in enumerate(model_data_pairs):
                    episode_reward[idx] += model.actor_data.rewards.reb_reward
                # track performance over episode
                for model in model_data_pairs:
                    episode_served_demand += model.actor_data.info.served_demand
                    episode_rebalancing_cost += model.actor_data.info.rebalancing_cost
                step += 1
                for model in model_data_pairs:
                    if i_episode > 10:
                        # sample from memory and update model
                        batch = model.model.replay_buffer.sample_batch(
                            config.batch_size, norm=False
                        )
                        model.model.update(data=batch)

                for model_data_pair in model_data_pairs:
                    model_data_pair.actor_data.model_log.reward += (
                        model_data_pair.actor_data.rewards.pax_reward
                    )
                    model_data_pair.actor_data.model_log.reward += (
                        model_data_pair.actor_data.rewards.reb_reward
                    )
                    model_data_pair.actor_data.model_log.served_demand += (
                        model_data_pair.actor_data.info.served_demand
                    )
                    model_data_pair.actor_data.model_log.rebalancing_cost += (
                        model_data_pair.actor_data.info.rebalancing_cost
                    )

            epochs.set_description(
                f"Episode {i_episode+1} | "
                f"Reward_0: {episode_reward[0]:.2f} | "
                f"Reward_1: {episode_reward[1]:.2f} | "
                f"ServedDemand: {episode_served_demand:.2f} | "
                f"Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if np.sum(episode_reward) >= best_reward:
                for model in model_data_pairs:
                    model.model.save_checkpoint(
                        path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_sample.pth"
                    )
                best_reward = np.sum(episode_reward)
                logging_dict.update({"Best Reward": best_reward})

            for model_data_pair in model_data_pairs:
                logging_dict.update(
                    model_data_pair.actor_data.model_log.dict(
                        model_data_pair.actor_data.name
                    )
                )
            wandb.log(logging_dict)
            # if i_episode % 10 == 0:
            #     (
            #         test_reward,
            #         test_served_demand,
            #         test_rebalancing_cost,
            #     ) = model.test_agent(
            #         1, env, config.cplex_path, parser=parser
            #     )
            #     if test_reward >= best_reward_test:
            #         best_reward_test = test_reward
            #         model.save_checkpoint(
            #             path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_test.pth"
            #         )
    # else:
    #     """Run main testing loop."""
    #     logger.info("Running main testing loop for SAC.")
    #     scenario = Scenario(
    #         json_file=f"data/scenario_{config.city}.json",
    #         demand_ratio=config.demand_ratio[config.city],
    #         json_hr=config.json_hr[config.city],
    #         sd=config.seed,
    #         json_tstep=config.json_tstep,
    #         tf=config.max_steps,
    #     )
    #     env = AMoD(scenario, beta=config.beta[config.city])
    #     parser = GNNParser(env, T=6, json_file=f"data/scenario_{config.city}.json")
    #     model = SAC(
    #         env=env,
    #         input_size=13,
    #         hidden_size=256,
    #         p_lr=1e-3,
    #         q_lr=1e-3,
    #         alpha=0.3,
    #         batch_size=100,
    #         use_automatic_entropy_tuning=False,
    #         critic_version=config.critic_version,
    #     ).to(config.device)
    #     print("load model")
    #     model.load_checkpoint(
    #         path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_test.pth"
    #     )
    #     test_episodes = config.max_episodes  # set max number of training episodes
    #     # T = config.max_steps  # set episode length
    #     epochs = trange(test_episodes)  # epoch iterator
    #     # Initialize lists for logging
    #     # log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}
    #     rewards = []
    #     demands = []
    #     costs = []
    #     for episode in range(10):
    #         episode_reward = 0
    #         episode_served_demand = 0
    #         episode_rebalancing_cost = 0
    #         obs = env.reset()
    #         done = False
    #         k = 0
    #         pax_reward = 0
    #         while not done:
    #             # take matching step (Step 1 in paper)
    #             actor_data, done = env.pax_step(
    #                 cplex_path=config.cplexpath,
    #                 path=config.path,
    #             )
    #             for model in models:
    #                 model.actor_data.model_log.reward += (
    #                     model.actor_data.rewards.pax_reward
    #                 )
    #             episode_reward += pax_reward
    #             pax_reward += pax_reward
    #             # use GNN-RL policy (Step 2 in paper)
    #             o = parser.parse_obs()
    #             action_rl = model.select_action(o, deterministic=True)
    #             # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
    #             desired_acc = {
    #                 env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
    #                 for i in range(len(env.region))
    #             }
    #             # solve minimum rebalancing distance problem (Step 3 in paper)
    #             reb_action = solveRebFlow(
    #                 env, config.path, desired_acc, config.cplexpath, config.directory
    #             )
    #             _, reb_reward, done, info, _, _ = env.reb_step(reb_action)
    #             episode_reward += reb_reward
    #             # track performance over episode
    #             episode_served_demand += info["served_demand"]
    #             episode_rebalancing_cost += info["rebalancing_cost"]
    #             k += 1
    #         # Send current statistics to screen
    #         epochs.set_description(
    #             f"Episode {episode + 1} | "
    #             f"Reward: {episode_reward:.2f} | "
    #             f"ServedDemand: {episode_served_demand:.2f} | "
    #             f"Reb. Cost: {episode_rebalancing_cost:.2f}"
    #         )
    #         # Log KPIs
    #         rewards.append(episode_reward)
    #         demands.append(episode_served_demand)
    #         costs.append(episode_rebalancing_cost)
    #     print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
    #     print("Served demand (mean, std):", np.mean(demands), np.std(demands))
    #     print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))


if __name__ == "__main__":
    city = City.brooklyn
    config = args_to_config(city)
    config.max_episodes = 1000
    main(config)
