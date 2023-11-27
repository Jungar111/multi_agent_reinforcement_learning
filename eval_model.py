"""Main file for testing model."""
import copy

import numpy as np
from tqdm import trange
import json
import pandas as pd

from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.algos.sac import SAC
from multi_agent_reinforcement_learning.algos.sac_gnn_parser import GNNParser
from multi_agent_reinforcement_learning.data_models.actor_data import (
    ActorData,
    ModelLog,
)
from multi_agent_reinforcement_learning.data_models.city_enum import City
from multi_agent_reinforcement_learning.data_models.config import SACConfig
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    plot_price_diff_over_time,
    plot_average_distribution,
)
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.sac_argument_parser import args_to_config

logger = init_logger()


def main(config: SACConfig):
    """Main loop for training and testing."""
    advesary_number_of_cars = int(config.total_number_of_cars / 2)
    actor_data = [
        ActorData(
            name="RL_1_SAC",
            no_cars=config.total_number_of_cars - advesary_number_of_cars,
        ),
        ActorData(name="RL_2_SAC", no_cars=advesary_number_of_cars),
    ]

    logger.info("Running main test loop.")
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
    test_episodes = config.max_episodes
    # T = config.max_steps
    epochs = trange(test_episodes)

    with open(str(env.config.json_file)) as file:
        data = json.load(file)

    df = pd.DataFrame(data["demand"])
    init_price_dict = df.groupby(["origin", "destination"]).price.mean().to_dict()
    init_price_mean = df.price.mean()

    for idx, model_data_pair in enumerate(model_data_pairs):
        logger.info(
            f"Loading from saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )
        model_data_pair.model.load_checkpoint(
            path=f"saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )

    epoch_prices = []

    for i_episode in epochs:
        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.model_log = ModelLog()

        all_actions = np.zeros(
            (
                len(model_data_pairs),
                config.tf,
                # nr of regions
                config.grid_size_x,
            )
        )

        env.reset(model_data_pairs)  # initialize environment
        episode_reward = [0, 0]
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        done = False
        step = 0
        o = [None, None]
        action_rl = [None, None]
        obs_list = [None, None]
        prices = {
            0: [],
            1: [],
        }
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
            for idx, model_data_pair in enumerate(model_data_pairs):
                o[idx] = parser.parse_obs(model_data_pair.actor_data.graph_state)
                episode_reward[idx] += model_data_pair.actor_data.rewards.pax_reward
                if step > 0:
                    # store transition in memory
                    rl_reward = (
                        model_data_pair.actor_data.rewards.pax_reward
                        + model_data_pair.actor_data.rewards.reb_reward
                    )
                    model_data_pair.model.replay_buffer.store(
                        obs_list[idx],
                        action_rl[idx],
                        config.rew_scale * rl_reward,
                        o[idx],
                    )
                action_rl[idx], price = model_data_pair.model.select_action(o[idx])
                for i in range(config.grid_size_x):
                    for j in range(config.grid_size_y):
                        model_data_pair.actor_data.graph_state.price[i, j][step + 1] = (
                            init_price_dict.get((i, j), init_price_mean) + price
                        )
                prices[idx].append(price)

            for idx, model in enumerate(model_data_pairs):
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                model.actor_data.flow.desired_acc = {
                    env.region[i]: int(
                        action_rl[idx][i]
                        * dictsum(model.actor_data.graph_state.acc, env.time + 1)
                    )
                    for i in range(len(env.region))
                }
                all_actions[idx, step, :] = list(
                    model_data_pairs[idx].actor_data.flow.desired_acc.values()
                )

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

            # track performance over episode
            for model_data_pair in model_data_pairs:
                reward_for_episode = (
                    model_data_pair.actor_data.rewards.pax_reward
                    + model_data_pair.actor_data.rewards.reb_reward
                )
                model_data_pair.actor_data.model_log.reward += reward_for_episode

                model_data_pair.actor_data.model_log.revenue_reward += (
                    model_data_pair.actor_data.rewards.pax_reward
                )
                model_data_pair.actor_data.model_log.rebalancing_reward += (
                    model_data_pair.actor_data.rewards.reb_reward
                )

                model_data_pair.actor_data.model_log.served_demand += (
                    model_data_pair.actor_data.info.served_demand
                )
                model_data_pair.actor_data.model_log.rebalancing_cost += (
                    model_data_pair.actor_data.info.rebalancing_cost
                )
                model_data_pair.model.rewards.append(reward_for_episode)
            step += 1

        epochs.set_description(
            f"Episode {i_episode+1} | "
            f"Reward_0: {episode_reward[0]:.2f} | "
            f"Reward_1: {episode_reward[1]:.2f} | "
            f"ServedDemand: {episode_served_demand:.2f} | "
            f"Reb. Cost: {episode_rebalancing_cost:.2f} | "
            f"Mean price: {np.mean(prices[0]):.2f}"
        )

        epoch_prices.append(prices)

    plot_price_diff_over_time(epoch_prices)

    plot_average_distribution(
        actions=np.array(all_actions),
        T=config.tf,
        model_data_pairs=model_data_pairs,
    )


if __name__ == "__main__":
    city = City.san_francisco
    config = args_to_config(city, cuda=True)
    # config.tf = 180
    config.max_episodes = 1
    config.grid_size_x = 10
    config.grid_size_y = 10
    config.total_number_of_cars = 374
    config.wandb_mode = "disabled"
    main(config)
