#Helper.py

#Contains utility functions for evaluation smoothing plotting loading


import numpy as np
import matplotlib.pyplot as plt
import csv


def evaluate(agent, eval_env, n_episodes=5):
    #Run n_episodes greedy episodes and return mean return

    returns = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # fully greedy during evaluation

    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)

    agent.epsilon = original_epsilon
    return np.mean(returns)


def smooth(returns, window=9):
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    return smoothed


def load_baseline(csv_path='BaselineDataCartPole.csv'):
    '''Load baseline learning curve from CSV file'''
    steps, returns = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row['env_step']))
            returns.append(float(row['Episode_Return_smooth']))
    return np.array(steps), np.array(returns)


class LearningCurvePlot:
    def __init__(self, title='Learning Curve'):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Environment Steps')
        self.ax.set_ylabel('Episode Return')
        self.ax.set_title(title)

    def add_curve(self, steps, returns, label=''):
        self.ax.plot(steps, returns, label=label)

    def add_hline(self, value, label=''):
        self.ax.axhline(y=value, linestyle='--', color='black', label=label)

    def set_ylim(self, low, high):
        self.ax.set_ylim(low, high)

    def save(self, filename):
        self.ax.legend()
        self.fig.savefig(filename)
        plt.close(self.fig)
        print(f'Saved: {filename}')


if __name__ == '__main__':
    # Test load_baseline
    try:
        steps, returns = load_baseline()
        print(f"Baseline loaded: {len(steps)} points")
        print(f"Steps range: {steps[0]:.0f} - {steps[-1]:.0f}")
        print(f"Returns range: {returns.min():.1f} - {returns.max():.1f}")
    except FileNotFoundError:
        print("Baseline CSV not found - place BaselineDataCartPole.csv in same folder")
