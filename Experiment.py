#Experiment.py

import numpy as np
import time
import argparse
import gymnasium as gym
from DQN import dqn
from Agent import DQNAgent
from Helper import LearningCurvePlot, load_baseline, smooth, evaluate


# ============================================================
# SHARED SETTINGS
# ============================================================

N_REPETITIONS   = 5
SMOOTHING       = 9
N_STEPS         = 1_000_000
EVAL_INTERVAL   = 2_500
N_EVAL_EPISODES = 5
GAMMA           = 0.99
NUM_ENVS        = 8

COMMON_PARAMS = dict(
    n_steps            = N_STEPS,
    eval_interval      = EVAL_INTERVAL,
    n_eval_episodes    = N_EVAL_EPISODES,
    gamma              = GAMMA,
    epsilon            = 1.0,
    epsilon_min        = 0.01,
    epsilon_decay      = 0.995,
    hidden_size        = 64,
    batch_size         = 64,
    update_every       = 1,
    target_update_freq = 100,
    buffer_size        = 10000,
    learning_rate      = 1e-3,
    num_envs           = NUM_ENVS,
)

#Params specifically for task 2.4
TASK24_PARAMS = dict(
    n_steps            = N_STEPS,
    eval_interval      = EVAL_INTERVAL,
    n_eval_episodes    = N_EVAL_EPISODES,
    gamma              = GAMMA,
    epsilon            = 1.0,
    epsilon_min        = 0.01,
    epsilon_decay      = 0.995,
    hidden_size        = 64,
    batch_size         = 64,
    update_every       = 4,
    target_update_freq = 100,
    buffer_size        = 50000,
    learning_rate      = 1e-3,
)



# HELPER


def average_over_repetitions(n_repetitions=5, smoothing_window=9, **kwargs):
    returns_over_repetitions = []
    now = time.time()

    for rep in range(n_repetitions):
        print(f'    Rep {rep+1}/{n_repetitions}')
        returns, steps = dqn(**kwargs)
        returns_over_repetitions.append(returns)

    print(f'    Setting took {(time.time()-now)/60:.1f} minutes')
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)

    if smoothing_window is not None:
        learning_curve = smooth(learning_curve, smoothing_window)
        steps = steps[:len(learning_curve)]

    return learning_curve, steps



# PER-EPISODE EPSILON DECAY — used for task 2.4


def dqn_per_episode_decay(n_steps=1_000_000, eval_interval=2500, n_eval_episodes=5,
        learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
        epsilon_decay=0.995, hidden_size=64, batch_size=64, update_every=4,
        target_update_freq=100, buffer_size=50000,
        use_target_network=False, use_replay_buffer=False, **kwargs):

    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim, action_dim=action_dim,
        learning_rate=learning_rate, gamma=gamma,
        epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
        hidden_size=hidden_size, batch_size=batch_size, update_every=update_every,
        target_update_freq=target_update_freq, buffer_size=buffer_size,
        use_target_network=use_target_network, use_replay_buffer=use_replay_buffer
    )

    eval_returns, eval_steps = [], []
    step = 0

    while step < n_steps:
        state, _ = env.reset()
        done = False
        while not done and step < n_steps:
            if step % eval_interval == 0:
                eval_returns.append(evaluate(agent, eval_env, n_eval_episodes))
                eval_steps.append(step)
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if use_replay_buffer:
                agent.store(state, action, reward, next_state, done)
                agent.update_network(step)
            else:
                agent.update(state, action, reward, next_state, done)
            state = next_state
            step += 1
        # epsilon decays per episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    env.close(); eval_env.close()
    return np.array(eval_returns), np.array(eval_steps)


def average_over_repetitions_per_episode(n_repetitions=5, smoothing_window=9, **kwargs):
    returns_over_repetitions = []
    now = time.time()

    for rep in range(n_repetitions):
        print(f'    Rep {rep+1}/{n_repetitions}')
        returns, steps = dqn_per_episode_decay(**kwargs)
        returns_over_repetitions.append(returns)

    print(f'    Setting took {(time.time()-now)/60:.1f} minutes')
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)

    if smoothing_window is not None:
        learning_curve = smooth(learning_curve, smoothing_window)
        steps = steps[:len(learning_curve)]

    return learning_curve, steps



# TASK 2.1


def task_2_1(baseline_steps, baseline_returns):
    print('Task 2.1: Naive DQN')

    naive_curve, naive_steps = average_over_repetitions(
        n_repetitions    = N_REPETITIONS,
        smoothing_window = SMOOTHING,
        use_target_network = False,
        use_replay_buffer  = False,
        **COMMON_PARAMS
    )

    Plot = LearningCurvePlot(title='Task 2.1: Naive DQN on CartPole')
    Plot.set_ylim(0, 520)
    Plot.add_curve(naive_steps, naive_curve, label='Naive DQN')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('naive_dqn.png')

    return naive_curve, naive_steps



# TASK 2.2


def task_2_2(baseline_steps, baseline_returns):
    print('\nTask 2.2: Ablation Study')

    print('  Ablation: learning rate')
    Plot = LearningCurvePlot(title='Ablation: Learning Rate')
    Plot.set_ylim(0, 520)
    for lr in [1e-4, 1e-3, 1e-2]:
        print(f'    lr = {lr}')
        curve, steps = average_over_repetitions(
            n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
            use_target_network=True, use_replay_buffer=True,
            **{**COMMON_PARAMS, 'learning_rate': lr}
        )
        Plot.add_curve(steps, curve, label=f'lr = {lr}')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_lr.png')

    print('  Ablation: network size')
    Plot = LearningCurvePlot(title='Ablation: Network Size')
    Plot.set_ylim(0, 520)
    for size in [32, 64, 128]:
        print(f'    hidden = {size}')
        curve, steps = average_over_repetitions(
            n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
            use_target_network=True, use_replay_buffer=True,
            **{**COMMON_PARAMS, 'hidden_size': size}
        )
        Plot.add_curve(steps, curve, label=f'hidden = {size}')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_network.png')

    print('  Ablation: update frequency')
    Plot = LearningCurvePlot(title='Ablation: Update Frequency')
    Plot.set_ylim(0, 520)
    for freq in [1, 4, 8]:
        print(f'    update_every = {freq}')
        curve, steps = average_over_repetitions(
            n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
            use_target_network=True, use_replay_buffer=True,
            **{**COMMON_PARAMS, 'update_every': freq}
        )
        Plot.add_curve(steps, curve, label=f'update every {freq} steps')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_update.png')

    print('  Ablation: exploration factor (fixed epsilon)')
    Plot = LearningCurvePlot(title='Ablation: Exploration Factor')
    Plot.set_ylim(0, 520)
    for eps in [0.1, 0.3, 0.5]:
        print(f'    epsilon = {eps}')
        curve, steps = average_over_repetitions(
            n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
            use_target_network=True, use_replay_buffer=True,
            **{**COMMON_PARAMS, 'epsilon': eps, 'epsilon_min': eps, 'epsilon_decay': 1.0}
        )
        Plot.add_curve(steps, curve, label=f'epsilon = {eps}')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_epsilon.png')



# TASK 2.4


def task_2_4(baseline_steps, baseline_returns):
    print('\nTask 2.4: All 4 configurations')

    print('  Naive DQN')
    naive_curve, naive_steps = average_over_repetitions_per_episode(
        n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
        use_target_network=False, use_replay_buffer=False, **TASK24_PARAMS
    )

    print('  Only TN')
    tn_curve, tn_steps = average_over_repetitions_per_episode(
        n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
        use_target_network=True, use_replay_buffer=False, **TASK24_PARAMS
    )

    print('  Only ER')
    er_curve, er_steps = average_over_repetitions_per_episode(
        n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
        use_target_network=False, use_replay_buffer=True, **TASK24_PARAMS
    )

    print('  TN + ER')
    tner_curve, tner_steps = average_over_repetitions_per_episode(
        n_repetitions=N_REPETITIONS, smoothing_window=SMOOTHING,
        use_target_network=True, use_replay_buffer=True, **TASK24_PARAMS
    )

    Plot = LearningCurvePlot(title='Task 2.4: Naive vs TN vs ER vs TN+ER')
    Plot.set_ylim(0, 520)
    Plot.add_curve(naive_steps, naive_curve, label='Naive')
    Plot.add_curve(tn_steps,    tn_curve,    label='Only TN')
    Plot.add_curve(er_steps,    er_curve,    label='Only ER')
    Plot.add_curve(tner_steps,  tner_curve,  label='TN + ER')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('configurations.png')


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', '2.1', '2.2', '2.4'],
                        help='Which task to run (default: all)')
    args = parser.parse_args()

    baseline_steps, baseline_returns = load_baseline()

    if args.task == '2.1':
        task_2_1(baseline_steps, baseline_returns)
    elif args.task == '2.2':
        task_2_2(baseline_steps, baseline_returns)
    elif args.task == '2.4':
        task_2_4(baseline_steps, baseline_returns)
    elif args.task == 'all':
        task_2_1(baseline_steps, baseline_returns)
        task_2_2(baseline_steps, baseline_returns)
        task_2_4(baseline_steps, baseline_returns)

    print('\nPlots saved:')
    print('  naive_dqn.png, ablation_lr.png, ablation_network.png,')
    print('  ablation_update.png, ablation_epsilon.png, configurations.png')


if __name__ == '__main__':
    main()