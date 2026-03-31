#Experiment.py

#Usage:
    #python Experiment.py            # run all tasks in sequence
    #python Experiment.py --task 2.1 # run Task 2.1 only
    #python Experiment.py --task 2.2 # run Task 2.2 only
    #python Experiment.py --task 2.4 # run Task 2.4 only


import numpy as np
import time
import argparse
from DQN import dqn
from Helper import LearningCurvePlot, load_baseline, smooth


# ============================================================
# SHARED SETTINGS
# ============================================================

N_REPETITIONS   = 5
SMOOTHING       = 9
N_STEPS         = 1_000_000
EVAL_INTERVAL   = 2_500
N_EVAL_EPISODES = 5
GAMMA           = 0.99
NUM_ENVS        = 8      # parallel environments for speedup

# Fixed best hyperparameters used across all tasks
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
    target_update_freq = 500,
    buffer_size        = 10000,
    learning_rate      = 1e-4,
    num_envs           = NUM_ENVS,
)


# ============================================================
# HELPER
# ============================================================

def average_over_repetitions(n_repetitions=5, smoothing_window=9, **kwargs):
    """
    Runs multiple repetitions of DQN and averages the learning curves.
    Same structure as Assignment 1 average_over_repetitions().
    """
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


# ============================================================
# TASK 2.1 — Naive DQN (no Target Network, no Experience Replay)
# ============================================================

def task_2_1(baseline_steps, baseline_returns):

    #Runs naive DQN (no TN, no ER)
    #Returns the curve and steps for reuse in Task 2.4

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


# ============================================================
# TASK 2.2 — Ablation Study
# ============================================================

def task_2_2(baseline_steps, baseline_returns):

    #Task 2.2, Ablation study
    
    print('\nTask 2.2: Ablation Study')

    # --- Learning rate ---
    print('  Ablation: learning rate')
    Plot = LearningCurvePlot(title='Ablation: Learning Rate')
    Plot.set_ylim(0, 520)
    for lr in [1e-4, 1e-3, 1e-2]:
        print(f'    lr = {lr}')
        curve, steps = average_over_repetitions(
            n_repetitions    = N_REPETITIONS,
            smoothing_window = SMOOTHING,
            use_target_network = True,
            use_replay_buffer  = True,
            **{**COMMON_PARAMS, 'learning_rate': lr}
        )
        Plot.add_curve(steps, curve, label=f'lr = {lr}')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_lr.png')

    # --- Network size ---
    print('  Ablation: network size')
    Plot = LearningCurvePlot(title='Ablation: Network Size')
    Plot.set_ylim(0, 520)
    for size in [32, 64, 128]:
        print(f'    hidden = {size}')
        curve, steps = average_over_repetitions(
            n_repetitions    = N_REPETITIONS,
            smoothing_window = SMOOTHING,
            use_target_network = True,
            use_replay_buffer  = True,
            **{**COMMON_PARAMS, 'hidden_size': size}
        )
        Plot.add_curve(steps, curve, label=f'hidden = {size}')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_network.png')

    # --- Update-to-data ratio ---
    print('  Ablation: update frequency')
    Plot = LearningCurvePlot(title='Ablation: Update Frequency')
    Plot.set_ylim(0, 520)
    for freq in [1, 4, 8]:
        print(f'    update_every = {freq}')
        curve, steps = average_over_repetitions(
            n_repetitions    = N_REPETITIONS,
            smoothing_window = SMOOTHING,
            use_target_network = True,
            use_replay_buffer  = True,
            **{**COMMON_PARAMS, 'update_every': freq}
        )
        Plot.add_curve(steps, curve, label=f'update every {freq} steps')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_update.png')

    # --- Exploration factor (fixed epsilon, no decay) ---
    print('  Ablation: exploration factor (fixed epsilon)')
    Plot = LearningCurvePlot(title='Ablation: Exploration Factor')
    Plot.set_ylim(0, 520)
    for eps in [0.1, 0.3, 0.5]:
        print(f'    epsilon = {eps}')
        curve, steps = average_over_repetitions(
            n_repetitions    = N_REPETITIONS,
            smoothing_window = SMOOTHING,
            use_target_network = True,
            use_replay_buffer  = True,
            **{**COMMON_PARAMS,
               'epsilon'      : eps,
               'epsilon_min'  : eps,   # keep fixed throughout
               'epsilon_decay': 1.0}   # no decay
        )
        Plot.add_curve(steps, curve, label=f'epsilon = {eps}')
    Plot.ax.plot(baseline_steps, baseline_returns, '--', color='black', label='Baseline')
    Plot.save('ablation_epsilon.png')


# ============================================================
# TASK 2.4 — Compare all 4 configurations
# ============================================================

def task_2_4(baseline_steps, baseline_returns, naive_curve=None, naive_steps=None):
    #Task 2.4, configurations comparison
    #If naive_curve is provided (from task_2_1), it is reused directly.

    print('\nTask 2.4: All 4 configurations')

    if naive_curve is None or naive_steps is None:
        print('  Naive DQN')
        naive_curve, naive_steps = average_over_repetitions(
            n_repetitions    = N_REPETITIONS,
            smoothing_window = SMOOTHING,
            use_target_network = False,
            use_replay_buffer  = False,
            **COMMON_PARAMS
        )

    print('  Only TN')
    tn_curve, tn_steps = average_over_repetitions(
        n_repetitions    = N_REPETITIONS,
        smoothing_window = SMOOTHING,
        use_target_network = True,
        use_replay_buffer  = False,
        **COMMON_PARAMS
    )

    print('  Only ER')
    er_curve, er_steps = average_over_repetitions(
        n_repetitions    = N_REPETITIONS,
        smoothing_window = SMOOTHING,
        use_target_network = False,
        use_replay_buffer  = True,
        **COMMON_PARAMS
    )

    print('  TN + ER')
    tner_curve, tner_steps = average_over_repetitions(
        n_repetitions    = N_REPETITIONS,
        smoothing_window = SMOOTHING,
        use_target_network = True,
        use_replay_buffer  = True,
        **COMMON_PARAMS
    )

    Plot = LearningCurvePlot(title='Task 2.4: Naive vs TN vs ER vs TN+ER')
    Plot.set_ylim(0, 520)
    Plot.add_curve(naive_steps,  naive_curve,  label='Naive')
    Plot.add_curve(tn_steps,     tn_curve,     label='Only TN')
    Plot.add_curve(er_steps,     er_curve,     label='Only ER')
    Plot.add_curve(tner_steps,   tner_curve,   label='TN + ER')
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
        naive_curve, naive_steps = task_2_1(baseline_steps, baseline_returns)
        task_2_2(baseline_steps, baseline_returns)
        task_2_4(baseline_steps, baseline_returns,
                 naive_curve=naive_curve, naive_steps=naive_steps)

    print('\nPlots saved:')
    print('  naive_dqn.png, ablation_lr.png, ablation_network.png,')
    print('  ablation_update.png, ablation_epsilon.png, configurations.png')


if __name__ == '__main__':
    main()
