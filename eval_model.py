"""Main file for testing model."""
import copy
import json
from collections import defaultdict

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
from multi_agent_reinforcement_learning.data_models.price_model import PriceModel
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    get_summary_stats,
    plot_actions_as_function_of_time,
    plot_average_distribution,
    plot_price_distribution,
    plot_price_over_time,
    plot_price_vs_other_attribute,
)
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.sac_argument_parser import args_to_config

logger = init_logger()


def main(config: SACConfig, run_name: str, price_model: PriceModel):
    """Train SAC algorithm."""
    number_of_cars = int(config.no_cars / config.no_actors)
    actor_data = [
        ActorData(
            name=f"RL_{i+1}_SAC",
            no_cars=number_of_cars,
        )
        for i in range(config.no_actors)
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

    rl_actors = [
        SAC(
            env=env,
            config=config,
            actor_data=actor_data[i],
            input_size=13,
            hidden_size=config.hidden_size,
            p_lr=config.p_lr,
            q_lr=config.q_lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            use_automatic_entropy_tuning=False,
            clip=config.clip,
        )
        for i in range(config.no_actors)
    ]
    model_data_pairs = [ModelDataPair(rl_actors[i], actor_data[i]) for i in range(config.no_actors)]

    test_episodes = config.max_episodes
    # T = config.max_steps
    epochs = trange(test_episodes)

    with open(str(env.config.json_file)) as file:
        data = json.load(file)

    df = pd.DataFrame(data["demand"])
    init_price_dict = df.groupby(["origin", "destination"]).price.mean().to_dict()
    init_price_mean = df.price.mean()
    df["converted_time_stamp"] = (df["time_stamp"] - config.json_hr[config.city] * 60) // config.json_tstep
    travel_time_dict = df.groupby(["origin", "destination", "converted_time_stamp"])["travel_time"].mean().to_dict()

    for idx, model_data_pair in enumerate(model_data_pairs):
        logger.info(f"Loading from saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth")
        model_data_pair.model.load_checkpoint(
            path=f"saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )

    epoch_prices = []
    # actions_over_epoch = []
    actions_over_epoch = np.zeros((2, config.n_regions[config.city], test_episodes, config.tf))
    epoch_served_demand = []
    epoch_cancelled_demand = []
    epoch_unmet_demand = []
    epoch_rewards = defaultdict(list)
    model_data_pair_prices = defaultdict(list)
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
        episode_reward = [0 for _ in range(config.no_actors)]
        episode_served_demand = defaultdict(list)
        episode_cancelled_demand = defaultdict(list)
        episode_unmet_demand = defaultdict(list)
        prices = defaultdict(list)
        episode_rebalancing_cost = 0
        done = False
        step = 0
        o = [None for _ in range(config.no_actors)]
        action_rl = [None for _ in range(config.no_actors)]
        obs_list = [None for _ in range(config.no_actors)]
        while not done:
            # take matching step (Step 1 in paper)
            if step > 0:
                for i in range(config.no_actors):
                    obs_list[i] = copy.deepcopy(o[i])

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
                        model_data_pair.actor_data.rewards.pax_reward + model_data_pair.actor_data.rewards.reb_reward
                    )
                    if config.include_price:
                        model_data_pair.model.replay_buffer.store(
                            obs_list[idx],
                            action_rl[idx],
                            config.rew_scale * rl_reward,
                            o[idx],
                            prices[idx],
                        )
                    else:
                        model_data_pair.model.replay_buffer.store(
                            obs_list[idx],
                            action_rl[idx],
                            config.rew_scale * rl_reward,
                            o[idx],
                            None,
                        )
                if config.include_price:
                    action_rl[idx], price = model_data_pair.model.select_action(o[idx], deterministic=True)

                    for i in range(config.n_regions[config.city]):
                        for j in range(config.n_regions[config.city]):
                            tt = travel_time_dict.get((i, j, step * config.json_tstep), 1)
                            model_data_pair.actor_data.flow.value_of_time[i, j][step + 1] = price[0][0]

                            model_data_pair.actor_data.flow.travel_time[i, j][step + 1] = tt

                            if price_model == PriceModel.REG_MODEL:
                                model_data_pair.actor_data.graph_state.price[i, j][step + 1] = max(
                                    (price[0][0] * tt), 10
                                )
                            elif price_model == PriceModel.DIFF_MODEL:
                                model_data_pair.actor_data.graph_state.price[i, j][step + 1] = (
                                    init_price_dict.get((i, j), init_price_mean) + price[0][0]
                                )
                            elif price_model == PriceModel.ZERO_DIFF_MODEL:
                                model_data_pair.actor_data.graph_state.price[i, j][step + 1] = init_price_dict.get(
                                    (i, j), init_price_mean
                                )
                    prices[idx].append(price[0][0])
                else:
                    action_rl[idx] = model_data_pair.model.select_action(o[idx], deterministic=True)
                    for i in range(config.n_regions[config.city]):
                        for j in range(config.n_regions[config.city]):
                            tt = travel_time_dict.get((i, j, step * config.json_tstep), 1)

                            model_data_pair.actor_data.flow.travel_time[i, j][step + 1] = tt

                # price = price[0][0] if config.include_price else 0
                actions_over_epoch[idx, :, i_episode, step] = action_rl[idx]

            for idx, model in enumerate(model_data_pairs):
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                model.actor_data.flow.desired_acc = {
                    env.region[i]: int(action_rl[idx][i] * dictsum(model.actor_data.graph_state.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                all_actions[idx, step, :] = list(model_data_pairs[idx].actor_data.flow.desired_acc.values())

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
                unmet_demand = sum(sum(inner_dict.values()) for inner_dict in model.actor_data.unmet_demand.values())
                episode_served_demand[idx].append(model.actor_data.info.served_demand)
                episode_cancelled_demand[idx].append(model.actor_data.model_log.bus_unmet_demand)
                episode_unmet_demand[idx].append(unmet_demand)
                episode_rebalancing_cost += model.actor_data.info.rebalancing_cost

            # track performance over episode
            for model_data_pair in model_data_pairs:
                reward_for_episode = (
                    model_data_pair.actor_data.rewards.pax_reward + model_data_pair.actor_data.rewards.reb_reward
                )
                model_data_pair.actor_data.model_log.reward += reward_for_episode

                model_data_pair.actor_data.model_log.revenue_reward += model_data_pair.actor_data.rewards.pax_reward
                model_data_pair.actor_data.model_log.rebalancing_reward += model_data_pair.actor_data.rewards.reb_reward

                model_data_pair.actor_data.model_log.served_demand += model_data_pair.actor_data.info.served_demand
                model_data_pair.actor_data.model_log.rebalancing_cost += (
                    model_data_pair.actor_data.info.rebalancing_cost
                )
                model_data_pair.model.rewards.append(reward_for_episode)
            step += 1

        description = f"Episode {i_episode+1} | " f"Reward 1: {episode_reward[0]:.2f} | "
        if config.no_actors > 1:
            description += f"Reward 2: {episode_reward[1]:.2f} |"
        epochs.set_description(description)

        for idx in range(len(model_data_pairs)):
            epoch_rewards[idx].append(episode_reward[idx])
            model_data_pair_prices[idx].append(model_data_pairs[idx])

        epoch_prices.append(prices)
        epoch_served_demand.append(episode_served_demand)
        epoch_cancelled_demand.append(episode_cancelled_demand)
        epoch_unmet_demand.append(episode_unmet_demand)

    get_summary_stats(
        epoch_prices,
        epoch_rewards,
        epoch_served_demand,
        epoch_cancelled_demand,
        epoch_unmet_demand,
        run_name,
        config,
    )

    if config.no_actors > 1:
        plot_price_over_time(epoch_prices, name=run_name)
        plot_price_distribution(model_data_pair_prices=model_data_pair_prices, data=df, name=run_name)

        plot_price_vs_other_attribute(epoch_prices, epoch_served_demand, "served_demand", plot_name=run_name)
        plot_price_vs_other_attribute(epoch_prices, epoch_unmet_demand, "unmet_demand", plot_name=run_name)

        plot_actions_as_function_of_time(
            actions=np.array(actions_over_epoch),
            chosen_areas=[8, 9, 10],
            colors=["#8C1C13", "#2F4550", "#A3BBAD"],
            name=run_name,
        )
        plot_average_distribution(
            config=config,
            actions=all_actions,
            T=20,
            model_data_pairs=model_data_pairs,
            name=run_name,
        )


if __name__ == "__main__":
    city = City.san_francisco
    config = args_to_config(city, cuda=True)
    config.tf = 20
    config.max_episodes = 10
    # config.total_number_of_cars = 374
    config.wandb_mode = "disabled"
    config.include_price = True
    config.no_actors = 2
    config.cancellation = True
    price_model = PriceModel.DIFF_MODEL
    main(config, run_name="SAC_2_actor_org_price_test", price_model=price_model)
