import math
import torch

    
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
        self.world = make_env(10, 10)

    @property
    def n_states(self):
        "Number of states"
        return self.world.size(0)

    @property
    def n_actions(self):
        "Number of actions"
        return self.world.size(1)
    
    def step(self, state, action):
        "Simulate a step in the environment"
        next_state_probs = self.world[state, action]
        next_state = torch.multinomial(next_state_probs, 1).item()
        return next_state


def entropies(self):
    "compute information content of the world: entropies of next state distributions for all state-action pairs"
    eps = 1e-8
    p = self.world
    return -(p * (p+eps).log()).sum(-1)


class Agent:
    def __init__(self, n_states, n_actions, gamma=0.9, lr=0.5, mu=0.1, reward_kind='ext'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.mu = mu
        self.reward_kind = reward_kind

        self.q = torch.zeros(n_states, n_actions)
        self.world = torch.zeros(n_states, n_actions, n_states)
        self.generator = torch.Generator().manual_seed(42)

    def reward(self, p_old, p_new):
        "Compute intrinsic reward given old and new probability distributions of the next state"
        match self.reward_kind:
            case 'prob_diff':
                reward = torch.sum(torch.abs(p_new - p_old), -1).item()
            case 'ent_diff':
                reward = torch.abs(torch.sum(torch.where(
                    (p_new == 0),
                    torch.zeros_like(p_new),
                    p_new * torch.log(p_new)
                ), -1) - torch.sum(torch.where(
                    (p_old == 0),
                    torch.zeros_like(p_old),
                    p_old * torch.log(p_old)
                ), -1)).item()
            case 'kl':
                reward = torch.where(
                    (p_new == 0)|(p_old == 0),
                    torch.zeros_like(p_new),
                    p_new * (p_new.log() - p_old.log())
                ).sum(-1).item()
            case _:
                raise ValueError(f'Invalid reward kind: {self.reward_kind}')
        return reward

    def policy(self, state):
        "Explore with probability mu, otherwise choose the action with the highest Q-value"
        if torch.rand(1, generator=self.generator).item() <= self.mu:
            return torch.randint(0, self.n_actions, (1,), generator=self.generator).item()
        else:
            return self.q[state].argmax().item()

    def update_world(self, state, action, next_state):
        "Update world model estimates"
        denom = self.world[state,action,:].sum()
        zeros = torch.zeros_like(denom)
        p_old = torch.where(denom == 0, zeros, self.world[state,action] / denom)

        self.world[state,action,next_state] += 1
        p_new = torch.where((denom+1) == 0, zeros, self.world[state,action] / (denom+1))

        return p_old, p_new

    def update(self, state, action, next_state, reward):
        "Update Q-values using the Q-learning update rule"
        best_next_action = self.q[next_state].argmax().item()
        td_target = reward + self.gamma * self.q[next_state,best_next_action]
        self.q[state,action] = (1 - self.lr) * self.q[state,action] + self.lr * td_target


def step(agent, env, state):
    action = agent.policy(state)
    next_state = env.step(state, action)

    p_old, p_new = agent.update_world(state, action, next_state)
    reward = agent.reward(p_old, p_new)
    agent.update(state, action, next_state, reward)

    return next_state


def fkl(p, q, eps=1e-8):
    return (p * (p+eps).log() - p * (q+eps).log()).sum(-1)

def rkl(p, q, eps=1e-8):
    return fkl(q, p)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gamma = 0.9  # Discount factor
    lr = 0.5  # Learning rate
    mu = 0.1  # Exploration probability

    env = Environment()
    max_ent = math.log(env.n_states)
    print(f'information content: {entropies(env).sum().item():.3f} nats, ', f'max entropy: {max_ent*100:.3f} nats')

    agents = [
        Agent(env.n_states, env.n_actions, gamma, lr, 1.0, 'prob_diff'),
        Agent(env.n_states, env.n_actions, gamma, lr, mu, 'prob_diff'),
        Agent(env.n_states, env.n_actions, gamma, lr, mu, 'ent_diff'),
        Agent(env.n_states, env.n_actions, gamma, lr, mu, 'kl'),
    ]
    print('step', *[agent.reward_kind for agent in agents])

    initial_state = 0
    states = [initial_state for _ in agents]

    with torch.inference_mode():
        for exp in range(1,2**18+1):
            kls = []
            for index, agent in enumerate(agents):
                states[index] = step(agent, env, states[index])
                kls.append(fkl(agent.world/agent.world.sum(-1, keepdim=True), env.world))

            if math.log2(exp) % 1 == 0:
                print(exp, *[f'{torch.nan_to_num(kld, nan=max_ent).sum().item():.3f}' for kld in kls])

                fig, axs = plt.subplots(1, len(agents)+1, figsize=(15, 5), sharey=True)
                axs[0].matshow(entropies(env))
                axs[0].set_title('true world entropies')
                for ax, agent, kld in zip(axs[1:], agents, kls):
                    #ax.matshow(entropies(agent))
                    ax.matshow(torch.nan_to_num(kld, nan=max_ent))
                    ax.set_title(agent.reward_kind + ' agent')
                plt.savefig('world.pdf')
                plt.close(fig)

                fig, axs = plt.subplots(1, len(agents), figsize=(15, 5))
                plt.suptitle('q')
                for i, (ax, agent) in enumerate(zip(axs, agents)):
                    ax.matshow(agent.q)
                    if i == 0:
                        ax.set_title(agent.reward_kind + ' random search')
                    else:
                        ax.set_title(agent.reward_kind)
                plt.savefig('q.pdf')
                plt.close(fig)