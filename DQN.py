import numpy as np
import gymnasium as gym
from Agent import DQNAgent
from Helper import evaluate


def dqn(n_steps=100000,
        eval_interval=2500,
        n_eval_episodes=5,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        hidden_size=64,
        batch_size=64,
        update_every=1,
        target_update_freq=500,
        buffer_size=10000,
        use_target_network=False,
        use_replay_buffer=False,
        num_envs=8):

    if use_replay_buffer:
        env = gym.make_vec('CartPole-v1', num_envs=num_envs)
    else:
        env = gym.make('CartPole-v1')

    eval_env = gym.make('CartPole-v1')

    if use_replay_buffer:
        state_dim  = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.n
    else:
        state_dim  = env.observation_space.shape[0]
        action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        hidden_size=hidden_size,
        batch_size=batch_size,
        update_every=update_every,
        target_update_freq=target_update_freq,
        buffer_size=buffer_size,
        use_target_network=use_target_network,
        use_replay_buffer=use_replay_buffer
    )

    eval_returns = []
    eval_steps   = []
    states, _    = env.reset()
    step         = 0

    while step < n_steps:

        if step % eval_interval == 0:
            mean_return = evaluate(agent, eval_env, n_eval_episodes)
            eval_returns.append(mean_return)
            eval_steps.append(step)

        if use_replay_buffer:
            actions = np.array([agent.select_action(s) for s in states])
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated

            for i in range(num_envs):
                agent.store(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]
                )

            agent.update_network(step)
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            states  = next_states
            step   += num_envs

        else:
            action = agent.select_action(states)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(states, action, reward, next_state, done)

            if done:
                states, _ = env.reset()
            else:
                states = next_state

            step += 1

    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_steps)