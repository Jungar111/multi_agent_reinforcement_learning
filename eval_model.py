"""Main file for testing model."""
import copy
import json

import numpy as np
import pandas as pd
from tqdm import trange

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
    plot_actions_as_function_of_time,
    plot_average_distribution,
    plot_price_diff_over_time,
    plot_price_distribution,
    plot_price_vs_other_attribute,
)
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.sac_argument_parser import args_to_config

logger = init_logger()


def main(config: SACConfig):
    """Train SAC algorithm."""
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
    # init_price_dict = df.groupby(["origin", "destination"]).price.mean().to_dict()
    # init_price_mean = df.price.mean()
    df["converted_time_stamp"] = (
        df["time_stamp"] - config.json_hr[config.city] * 60
    ) // config.json_tstep
    travel_time_dict = (
        df.groupby(["origin", "destination", "converted_time_stamp"])["travel_time"]
        .mean()
        .to_dict()
    )

    for idx, model_data_pair in enumerate(model_data_pairs):
        logger.info(
            f"Loading from saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )
        model_data_pair.model.load_checkpoint(
            path=f"saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )

    epoch_prices = []
    # actions_over_epoch = []
    actions_over_epoch = np.zeros(
        (2, config.n_regions[config.city], test_episodes, config.tf)
    )
    epoch_served_demand = []
    epoch_unmet_demand = []
    for i_episode in epochs:
        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.model_log = ModelLog()

        all_actions = np.zeros(
            (
                len(model_data_pairs),
                config.tf,
                config.n_regions[config.city],
            )
        )

        env.reset(model_data_pairs)  # initialize environment
        episode_reward = [0, 0]
        episode_served_demand = {0: [], 1: []}
        episode_unmet_demand = {0: [], 1: []}
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
                        prices[idx],
                    )
                action_rl[idx], price = model_data_pair.model.select_action(o[idx])
                for i in range(config.n_regions[config.city]):
                    for j in range(config.n_regions[config.city]):
                        tt = travel_time_dict.get((i, j, step * config.json_tstep), 0)
                        model_data_pair.actor_data.flow.value_of_time[i, j][
                            step + 1
                        ] = price[0][0]
                        model_data_pair.actor_data.graph_state.price[i, j][step + 1] = (
                            price[0][0] * tt
                        )

                        # model_data_pair.actor_data.graph_state.price[i, j][step + 1] = (
                        #     init_price_dict.get((i, j), init_price_mean) + price[0][0]
                        # )
                prices[idx].append(price[0][0])
                actions_over_epoch[idx, :, i_episode, step] = action_rl[idx]

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
            for idx, model in enumerate(model_data_pairs):
                unmet_demand = sum(
                    sum(inner_dict.values())
                    for inner_dict in model.actor_data.unmet_demand.values()
                )
                episode_served_demand[idx].append(model.actor_data.info.served_demand)
                episode_unmet_demand[idx].append(unmet_demand)
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
            f"ServedDemand: {np.sum([served_demand for served_demand in episode_served_demand.values()]):.2f} | "
            f"Reb. Cost: {episode_rebalancing_cost:.2f} | "
            f"Mean price: {np.mean(prices[0]):.2f}"
        )
        epoch_prices.append(prices)
        epoch_served_demand.append(episode_served_demand)
        epoch_unmet_demand.append(episode_unmet_demand)

    plot_price_diff_over_time(epoch_prices)
    plot_price_distribution(model_data_pairs=model_data_pairs, data=df)

    plot_price_vs_other_attribute(epoch_prices, epoch_served_demand, "served_demand")
    plot_price_vs_other_attribute(epoch_prices, epoch_unmet_demand, "unmet_demand")

    plot_actions_as_function_of_time(
        actions=np.array(actions_over_epoch),
        chosen_areas=[8, 9, 10],
        colors=["#8C1C13", "#2F4550", "#A3BBAD"],
    )
    plot_average_distribution(
        config=config, actions=all_actions, T=20, model_data_pairs=model_data_pairs
    )


if __name__ == "__main__":
    city = City.san_francisco
    config = args_to_config(city, cuda=True)
    config.tf = 20
    config.max_episodes = 10
    config.total_number_of_cars = 374
    config.wandb_mode = "disabled"
    main(config)
