import warnings
warnings.filterwarnings("ignore")
import numpy as np
import wandb
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch as T
from utils import Actor, Critic, worker, test_episode


LOGS = True

if LOGS:
    import wandb
    wandb.init(project="lunar_lander_ac")

args = {'obs_space': 8,
        'action_space': 2,
        'n_hidden': 128,
        'n_steps': 10,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'device': 'cuda:0',
        'gamma': .95,
        'entropy_beta': 0.001,
        'critic_weight': 1,
        'workers': 15,
        'eps_per_worker': 250}

if __name__ == '__main__':

    models = {'actor': Actor(args['obs_space'], args['n_hidden'], args['action_space'], args['lr_actor'], args['device']),
              'critic': Critic(args['obs_space'], args['n_hidden'], 1, args['lr_critic'], args['device'])}
    models['actor'].share_memory()
    models['critic'].share_memory()

    #wandb seems to not like tracking models with shared parameters
    # if LOGS:
    #     wandb.watch(models['actor'])
    #     wandb.watch(models['critic'])

    mean_rewards = []
    episodes = 0

    processes = []
    total_runs = mp.Value('i', 0)

    rewards = mp.Queue()

    while True:
        for i in range(args['workers']):
            p = mp.Process(target=worker, args=(i, total_runs, models, args))
            p.start()
            processes.append(p)
        for p in processes:
                p.join()
        for p in processes:
                p.terminate()

        rewards = test_episode(models['actor'])
        mean_rewards.append(rewards)
        episodes += 1
        if LOGS:
            wandb.log({'rewards': rewards})
        print(f'Mean Reward: {np.mean(np.array(mean_rewards[-10:])):.2f} after'
              f' {episodes* args["workers"] * args["eps_per_worker"]} runs')

        if np.mean(np.array(mean_rewards[-10:])) > 200.:
            print("SOLVED!")
            T.save(models['actor'], 'actor_solved.h5')
            T.save(models['critic'], 'critic_solved.h5')
            break










