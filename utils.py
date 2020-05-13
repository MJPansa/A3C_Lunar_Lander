import gym
import torch as T
import torch.nn.modules as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist



class Actor(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out, lr, device):
        super(Actor, self).__init__()
        self.device = device
        self.l1 = nn.Linear(n_inputs, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_hidden)
        self.mu = nn.Linear(n_hidden, n_out)
        self.sigma = nn.Linear(n_hidden, n_out)

        self.optimizer = optim.SGD(self.parameters(), lr)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).to(self.device)

        if not self.training:
            x = x.unsqueeze(0)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu((self.l3(x)))
        mu = T.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        sigma = T.clamp(sigma, 1e-4, 2)

        return mu, sigma


class Critic(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out, lr, device):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(n_inputs, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_hidden)
        self.l4 = nn.Linear(n_hidden, n_out)

        self.optimizer = optim.SGD(self.parameters(), lr)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).to(self.device)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        state_value = self.l4(x)

        return state_value


def play_episode(env, state, n_steps, actor):
    actor.eval()
    states = []
    actions = []
    rewards = []
    done = False

    for i in range(n_steps):
        mus, sigmas = actor(state)
        normal_dist = dist.Normal(mus, sigmas)
        action = normal_dist.sample()
        action = T.clamp(action, -1., 1.).squeeze()
        next_state, reward, done, _ = env.step(action.detach().cpu().numpy())

        states.append(T.Tensor(state).to(actor.device))
        actions.append(action)
        rewards.append(reward)

        if done:
            next_state = env.reset()
            return states, actions, rewards, done, next_state

        state = next_state.copy()

    return states, actions, rewards, done, next_state


def calculate_rewards(states, rewards, critic, gamma, n_steps):
    critic.eval()
    if len(rewards) < n_steps:
        state_value = 0.
    else:
        state_value = critic(states[-1]).detach().cpu().item()
    G = []
    for reward in reversed(rewards):
        state_value = state_value * gamma + reward
        G.append(state_value)
    return list(reversed(G))


def train_on_episode(states, actions, rewards, actor, critic, entropy_beta, critic_weight, device):
    actor.train()
    critic.train()
    actor.optimizer.zero_grad()
    critic.optimizer.zero_grad()

    states = T.stack(states).to(device)
    actions = T.stack(actions).to(device)

    mus, sigmas = actor(states)
    state_values = critic(states).squeeze()
    rewards = T.Tensor(rewards).to(device)
    normal_dist = dist.Normal(mus, sigmas)
    log_probs = normal_dist.log_prob(actions).mean(1)
    entropy = normal_dist.entropy()
    entropy = entropy.mean(1)
    entropy_loss = (-1 * entropy * entropy_beta).sum()
    critic_loss = T.pow((rewards - state_values), 2).sum()
    critic_loss = critic_loss * critic_weight
    actor_loss = (-1 * (log_probs * (rewards - state_values.detach()))).sum() + entropy_loss
    loss = critic_loss + actor_loss

    loss.backward()
    T.nn.utils.clip_grad_norm_(actor.parameters(), .3)
    T.nn.utils.clip_grad_norm_(critic.parameters(), 1)
    actor.optimizer.step()
    critic.optimizer.step()

    return [x.detach().cpu().numpy() for x in [actor_loss, critic_loss, loss, entropy_loss]]


def test_episode(actor):
    rewards = []
    env = gym.make('LunarLanderContinuous-v2')
    state = env.reset()
    done = False

    while not done:
        mus, sigmas = actor(state)
        normal_dist = dist.Normal(mus, sigmas)
        action = normal_dist.sample()
        action = T.clamp(action, -1., 1.)

        next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
        rewards.append(reward)
        state = next_state.copy()

    return sum(rewards)


def worker(id_n, total_runs, models, args):
    env = gym.make('LunarLanderContinuous-v2')
    state = env.reset()
    for n in range(args['eps_per_worker']):
        states, actions, rewards, done, next_state = play_episode(env, state, args['n_steps'], models['actor'])
        rewards = calculate_rewards(states, rewards, models['critic'], args['gamma'], args['n_steps'])
        actor_loss, critic_loss, loss, entropy_loss = train_on_episode(states, actions, rewards, models['actor'],
                                                                       models['critic'], args['entropy_beta'],
                                                                       args['critic_weight'], args['device'])
        state = next_state.copy()

        total_runs.value += 1
        # if total_runs.value % 100 == 0:
        #     print(f'worker ({id_n}), n_step_updates: {total_runs.value}')
