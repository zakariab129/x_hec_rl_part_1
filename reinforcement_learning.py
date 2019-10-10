import gym
import numpy as np
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

# import seaborn as sns


def print_frames(frames_to_print):

    for i, frame in enumerate(frames_to_print):

        clear_output(wait=True)
        # print(frame['frame'].getvalue())
        print(frame["frame"])
        print(f"Episode: {frame['episode']}")
        print(f"Number Actions: {frame['number_actions']}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Action Reward: {frame['reward']}")
        print(f"Total Reward: {frame['total_reward']}")
        sleep(0.05)


'''
def plot_sparsity_heatmap(q_table):
    sparsity_matrix = np.where(q_table == 0, 0, 1)
    f, ax = plt.subplots(figsize=(7, 8))
    sns.set(font_scale=1.25)
    heatmap = sns.heatmap(sparsity_matrix, cbar=False)
    plt.show()
'''

def moving_average(values, window):
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plot_steps_per_episode(number_steps_per_episode, smoothing=8):
    f1 = plt.figure(1, figsize=(12, 7))
    plt.title(" Title : Agent's number of actions per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Number of actions")
    plt.plot(number_steps_per_episode, label="raw plot")
    plt.plot(
        moving_average(number_steps_per_episode, smoothing),
        label="moving average",
        color="r",
    )
    plt.legend()
    f1.show()


def plot_average_reward(average_reward_per_episode, smoothing=10):
    f2 = plt.figure(2, figsize=(12, 7))
    plt.title(" Title : Agent's average reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average reward")
    plt.plot(average_reward_per_episode, label="raw plot")
    plt.plot(
        moving_average(average_reward_per_episode, smoothing),
        label="moving average",
        color="r",
    )
    plt.legend()
    f2.show()


def plot_maximum_q_value(maximum_q_value_per_episode, smoothing=8):
    f3 = plt.figure(3, figsize=(12, 7))
    plt.title(" Title : Agent's maximum q-value per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Max q-value")
    plt.plot(maximum_q_value_per_episode, label="raw plot")
    plt.plot(
        moving_average(maximum_q_value_per_episode, smoothing),
        label="moving average",
        color="r",
    )
    plt.legend()
    f3.show()


def plot_total_reward(total_reward_per_episode, smoothing=10):
    f4 = plt.figure(4, figsize=(12, 7))
    plt.title(" Title : Agent's total reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.plot(total_reward_per_episode, label="raw plot")
    plt.plot(
        moving_average(total_reward_per_episode, smoothing),
        label="moving average",
        color="r",
    )
    plt.legend()
    f4.show()


def plot_episode_duration(episodes_durations, smoothing=10):

    f5 = plt.figure(5, figsize=(12, 7))
    plt.title(" Title : Duration of each episode")
    plt.xlabel("Episodes")
    plt.ylabel("Duration(miliseconds)")
    plt.plot(episodes_durations, label="raw plot")
    plt.plot(moving_average(episodes_durations, 10), label="moving average", color="r")
    plt.legend()
    ax = plt.axes()
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x * 1000))
    ax.yaxis.set_major_formatter(ticks_y)
    f5.show()


def plot_performance_info(
    number_steps_per_episode,
    average_reward_per_episode,
    maximum_q_value_per_episode,
    total_reward_per_episode,
    episodes_durations,
    q_table,
):

    plot_steps_per_episode(number_steps_per_episode)
    plot_average_reward(average_reward_per_episode)
    plot_maximum_q_value(maximum_q_value_per_episode)
    plot_total_reward(total_reward_per_episode)
    plot_episode_duration(episodes_durations)
    # plot_sparsity_heatmap(q_table)


def select_epsilon_greedy_action(environment, q_table, current_state, epsilon=0):

    if (
        np.random.uniform(0, 1) < epsilon
    ):  # Will never be the case if the we are using pure random
        selected_action = np.argmax(q_table[current_state])  # Exploit current policy
    else:
        selected_action = environment.action_space.sample()  # Explore environment

    return selected_action


def select_softmax_action(environment, q_table, current_state, temperature=1):

    possible_actions_q_values = q_table[current_state]
    # To avoid overflow we alter the q-values by substracting the exponential of the highest q-value
    # By altering all q-values, the softmax remains the same
    altered_possible_actions_q_values = possible_actions_q_values - max(
        possible_actions_q_values
    )
    if temperature > 0:
        numerators = np.exp(altered_possible_actions_q_values / temperature)
        possible_actions_probabilites = numerators / numerators.sum()
        # Action will be selected given the softmax distribution probabilty
        selected_action = np.random.choice(
            np.arange(q_table.shape[1]), 1, p=possible_actions_probabilites
        )[0]
    else:
        # Since temperature is null, we end up in the same situation as a pure greedy approach
        selected_action = select_epsilon_greedy_action(
            environment, q_table, current_state, epsilon=temperature
        )

    return selected_action


def select_action_with_strategy(
    environment, q_table, current_state, strategy="greedy", exploration_factor=0.1
):
    if strategy == "greedy":
        selected_action = select_epsilon_greedy_action(
            environment=environment,
            q_table=q_table,
            current_state=current_state,
            epsilon=exploration_factor,
        )
    elif strategy == "softmax":
        selected_action = select_softmax_action(
            environment=environment,
            q_table=q_table,
            current_state=current_state,
            temperature=exploration_factor,
        )

    try:
        next_state, reward, done, info = environment.step(selected_action)
        return next_state, reward, done, info, selected_action
    except:
        print(
            "Invalid strategy choice, available strategies are either greedy or softmax"
        )


def update_q_table_with_TD(
    q_table, current_state, selected_action, reward, next_state, alpha, gamma
):

    old_q_value = q_table[current_state, selected_action]
    value_next_state = np.max(q_table[next_state])
    # Compute new Q-value following the Bellman Equation & Temporal Difference
    # If alpha was null there would be no temporal difference, which means pure exploitation
    new_q_value = (1 - alpha) * old_q_value + alpha * (
        reward + gamma * value_next_state
    )
    q_table[current_state, selected_action] = new_q_value

    return q_table


def decay_exploration_factor(exploration_factor, exploration_decay):
    new_exploration_factor = exploration_factor - exploration_decay
    return new_exploration_factor


def decay_learning_rate(learning_rate, learning_decay):
    new_learning_rate = learning_rate - learning_decay
    return new_learning_rate


def check_sanity_learning_rate(alpha):
    if alpha < 0 or alpha > 1:
        print(
            "Invalid value for the learning rate, please choose a value between 0 and 1"
        )
        return False
    else:
        return True


def check_sanity_discount_factor(gamma):
    if gamma < 0 or gamma > 1:
        print(
            "Invalid value for the discount factor, please choose a value between 0 and 1"
        )
        return False
    else:
        return True


def check_sanity_exploration_factor(exploration_factor, strategy):
    if strategy == "greedy" and (exploration_factor < 0 or exploration_factor > 1):
        print(
            "Invalid value, in epsilon-greedy strategy, epsilon must be between 0 and 1"
        )
        return False

    elif strategy == "softmax" and exploration_factor < 0:
        print("Invalid value for the temperature, it must not be negative")
        return False
    else:
        return True


def loop_actions_within_episode(
    environment,
    q_table,
    frames_to_print,
    alpha,
    gamma,
    exploration_factor,
    i,
    second_frame,
    third_frame,
    number_episodes,
    exploration_decay,
    learning_decay,
    strategy="greedy",
):

    current_state = environment.reset()
    nb_steps, reward = 0, 0
    done = False
    rewards_per_episode = []
    episode_start_time = time.time()
    if not check_sanity_discount_factor(gamma=gamma):
        return None
    elif not check_sanity_learning_rate(alpha=alpha):
        return None
    elif not check_sanity_exploration_factor(
        exploration_factor=exploration_factor, strategy=strategy
    ):
        return None
    else:
        # Loop of the episode
        while not done:
            next_state, reward, done, info, selected_action = select_action_with_strategy(
                environment, q_table, current_state, strategy, exploration_factor
            )

            rewards_per_episode.append(reward)
            if i in [0, second_frame, third_frame, number_episodes - 3]:
                frames_to_print.append(
                    {
                        "frame": environment.render(mode="ansi"),
                        "episode": i,
                        "state": current_state,
                        "action": selected_action,
                        "number_actions": nb_steps,
                        "reward": reward,
                        "total_reward": sum(rewards_per_episode),
                    }
                )
            # Update the Q-table
            q_table = update_q_table_with_TD(
                q_table=q_table,
                current_state=current_state,
                selected_action=selected_action,
                reward=reward,
                next_state=next_state,
                alpha=alpha,
                gamma=gamma,
            )
            current_state = next_state
            nb_steps += 1

        # Decay if necessary
        exploration_factor = decay_exploration_factor(
            exploration_factor, exploration_decay
        )
        alpha = decay_learning_rate(alpha, learning_decay)

        episode_stop_time = time.time()
        episode_duration = episode_stop_time - episode_start_time

        return (
            q_table,
            frames_to_print,
            episode_duration,
            nb_steps,
            rewards_per_episode,
            exploration_factor,
            alpha,
        )


def train_agent(
    environment,
    strategy="greedy",
    alpha=0.75,
    gamma=0.9,
    exploration_factor=1,
    number_episodes=700,
    exploration_decay=0,
    learning_decay=0,
):

    number_possible_states = environment.observation_space.n
    number_possible_actions = environment.action_space.n
    q_table = np.zeros([number_possible_states, number_possible_actions])
    frames_to_print, steps_within_episode, average_reward_per_episode, maximum_q_value_per_episode, total_reward_per_episode, episodes_durations = (
        [] for i in range(6)
    )
    second_frame = np.random.randint(
        np.round(0.25 * number_episodes), np.round(0.45 * number_episodes)
    )
    third_frame = np.random.randint(
        np.round(0.65 * number_episodes), np.round(0.85 * number_episodes)
    )
    for i in range(number_episodes):
        if i in np.arange(0, number_episodes, 0.1 * number_episodes):
            print("Episode :", i)
            print("Exploration Factor :", exploration_factor)
            print("Learning Rate :", alpha)
        # Performs actions and updates Q-table until the episode is finished
        q_table, frames_to_print, episode_duration, nb_steps, rewards_per_episode, exploration_factor, alpha = loop_actions_within_episode(
            environment,
            q_table,
            frames_to_print,
            alpha,
            gamma,
            exploration_factor,
            i,
            second_frame,
            third_frame,
            number_episodes,
            exploration_decay,
            learning_decay,
            strategy,
        )
        # Collect information about the episode
        episodes_durations.append(episode_duration)
        steps_within_episode.append(nb_steps)
        average_reward_per_episode.append(np.mean(rewards_per_episode))
        maximum_q_value_per_episode.append(np.max(q_table))
        total_reward_per_episode.append(np.sum(rewards_per_episode))

    print("Training finished.\n")
    return (
        q_table,
        frames_to_print,
        steps_within_episode,
        average_reward_per_episode,
        maximum_q_value_per_episode,
        total_reward_per_episode,
        episodes_durations,
    )
