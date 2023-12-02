import torch
import random

    
def make_env(n, m):
    "Create a transition matrix for the environment with n states and m actions"
    transition_matrix = torch.zeros(n, m, n)

    for i in range(n-1):
        for j in range(m-1):
            transition_matrix[i, j, i] = 1.
        j = m - 1
        transition_matrix[i, j, i+1] = 1.

    i = n - 1 # state that allows to acquire a lot of information
    for j in range(m):
        for k in range(n):
            transition_matrix[i, j, k] = 1/n

    return transition_matrix


class Environment:
    def __init__(self):
        self.transitions = make_env(10, 10)

    @property
    def n_states(self):
        "Number of states"
        return self.transitions.size(0)

    @property
    def n_actions(self):
        "Number of actions"
        return self.transitions.size(1)
    
    def step(self, state, action):
        "Simulate a step in the environment"
        next_state_probs = self.transitions[state, action]
        next_state = torch.multinomial(next_state_probs, 1).item()
        reward = 1
        return next_state, reward


class Agent:
    def __init__(self, n_states, n_actions, gamma, lr, mu):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.mu = mu
        self.q = torch.zeros(n_states, n_actions)
        self.counter = torch.zeros(n_states, n_actions, n_states)

    def policy(self, state):
        "Explore with probability mu, otherwise choose the action with the highest Q-value"
        if random.random() <= self.mu:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.q[state].argmax().item()

    def update_world(self, state, action, next_state):
        "Update world model estimates"
        denom = self.counter[state,action,:].sum()
        zeros = torch.zeros_like(denom)
        p_prev = torch.where(denom == 0, zeros, self.counter[state,action] / denom)

        self.counter[state,action,next_state] += 1
        p_next = torch.where((denom+1) == 0, zeros, self.counter[state,action] / (denom+1))

        return p_prev, p_next

    def update(self, state, action, next_state, reward):
        "Update Q-values using the Q-learning update rule"
        best_next_action = self.q[next_state].argmax().item()
        td_target = reward + self.gamma * self.q[next_state,best_next_action]
        self.q[state,action] = (1 - self.lr) * self.q[state,action] + self.lr * td_target


def train(agent, env, reward_kind='env'):
    state = random.randint(0, env.n_states - 1)
    while True:
        action = agent.policy(state)
        next_state, env_reward = env.step(state, action)

        p_prev, p_next = agent.update_world(state, action, next_state)
        match reward_kind:
            case 'env':
                reward = env_reward
            case 'prob_diff':
                reward = torch.sum(torch.abs(p_next - p_prev), -1).item()
                # if reward != 0.0:
                #     print(p_prev, p_next)
                #     print(state, action, next_state, reward)
            case 'ent_diff':
                reward = torch.abs(torch.sum(torch.where(
                    (p_next == 0),
                    torch.zeros_like(p_next),
                    p_next * torch.log(p_next)
                ), -1) - torch.sum(torch.where(
                    (p_prev == 0),
                    torch.zeros_like(p_prev),
                    p_prev * torch.log(p_prev)
                ), -1)).item()
            case 'kl':
                reward = torch.where(
                    (p_next == 0)|(p_prev == 0),
                    torch.zeros_like(p_next),
                    p_next * (p_next.log() - p_prev.log())
                ).sum(-1).item()
                #if reward != 0:
                #    print(state, action, next_state, reward)
            case _:
                raise ValueError(f'Invalid reward kind: {reward_kind}')

        agent.update(state, action, next_state, reward)
        state = next_state
        if state == env.n_states - 1:
            break


if __name__ == "__main__":
    gamma = 0.9  # Discount factor
    lr = 0.5  # Learning rate
    mu = 0.1  # Exploration probability

    env = Environment()
    agent = Agent(env.n_states, env.n_actions, gamma, lr, mu)

    num_experiments = 8192
    for exp in range(num_experiments):
        train(agent, env)
        if exp % 512 == 0:
            print(exp)

    # Print the learned Q-values
    print(agent.q)
