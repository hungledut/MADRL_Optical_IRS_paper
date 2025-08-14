import time

import torch

import numpy as np
import gymnasium as gym

# import our customed uav-environment and policy
from environment import IRS_env
from PPO_model_CNN import PPO
from arguments import parse_args


import matplotlib.pyplot as plt

import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
args = parse_args()

has_continuous_action_space = False

average_episode = 50            # how many average episodes that you want to print 
max_ep_len = args.max_step                 # max timesteps in one episode
max_training_timesteps = int(20e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
action_std = None

################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

time_step = 0
i_episode = 0

# initialize a PPO agent
ppo_agent0 = PPO(209, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent1 = PPO(209, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent2 = PPO(209, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


if __name__ == '__main__':

    env = IRS_env(L=args.L, lambda_=args.lambda_, h_UAV=args.h_UAV, h_HAP=args.h_HAP, zenith_angle=args.zenith_angle, 
                  theta_i=args.theta_i, wo=args.wo, a=args.a, users=args.users, uavs=args.uavs, size=args.size,
                  varphi_=args.varphi_, v0=args.v0, tau=args.tau, noise_power_FSO=args.noise_power_FSO, P_FSO=args.P_FSO, 
                  B_FSO=args.B_FSO, noise_power=args.noise_power, B_RF=args.B_RF, r_th=args.r_th, max_step= args.max_step,
                  grid_num=args.grid_num, Nc=args.Nc, Hcl=args.Hcl)
    # training loop
    rewards_per_episode = []
    percentage_user_episode = []

    while time_step <= max_training_timesteps:
        O_UAV0, O_UAV1, O_UAV2, CLWC = env.reset()
        S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = 0,0,0,0
        g_t = 0
        l_t_0 = 0
        l_t_1 = 0
        l_t_2 = 0
        w = 0.6
        percentage_users = []
        current_ep_reward = 0

        start = time.time()

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action0 = ppo_agent0.select_action(O_UAV0, CLWC)
            action1 = ppo_agent1.select_action(O_UAV1, CLWC)
            action2 = ppo_agent2.select_action(O_UAV2, CLWC)

            O_UAV0, O_UAV1, O_UAV2, CLWC, S, N_UAV0, N_UAV1, N_UAV2, _ = env.step([action0, action1, action2])
            if S > S_t:
                g_t = 1
            elif S < S_t:
                g_t = -1
            else:
                g_t = 0

            if N_UAV0 > N_UAV0_t:
                l_t_0 = 1
            elif N_UAV0 < N_UAV0_t:
                l_t_0 = -1
            else:
                l_t_0 = 0

            if N_UAV1 > N_UAV1_t:
                l_t_1 = 1
            elif N_UAV1 < N_UAV1_t:
                l_t_1 = -1
            else:
                l_t_1 = 0

            if N_UAV2 > N_UAV2_t:
                l_t_2 = 1
            elif N_UAV2 < N_UAV2_t:
                l_t_2 = -1
            else:
                l_t_2 = 0

            S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = S, N_UAV0, N_UAV1, N_UAV2

            # print(S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t)

            reward0 = w*l_t_0 + (1-w)*g_t
            reward1 = w*l_t_1 + (1-w)*g_t
            reward2 = w*l_t_2 + (1-w)*g_t

            # saving reward and is_terminals
            ppo_agent0.buffer.rewards.append(reward0)
            ppo_agent1.buffer.rewards.append(reward1)
            ppo_agent2.buffer.rewards.append(reward2)

            ppo_agent0.buffer.is_terminals.append(False)
            ppo_agent1.buffer.is_terminals.append(False)
            ppo_agent2.buffer.is_terminals.append(False)

            time_step +=1
            current_ep_reward += reward0
            current_ep_reward += reward1
            current_ep_reward += reward2

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent0.update()
                ppo_agent1.update()
                ppo_agent2.update()
            percentage_users.append(S_t*100/args.users)

        end = time.time()
        print('computational time =', end-start)

        rewards_per_episode.append(current_ep_reward)
        percentage_user_episode.append(S_t*100/args.users)
        print('Episode ', i_episode, ': Reward = ', current_ep_reward ,'The percentage of satisfied users = ', S_t*100/args.users , '%')
        if time_step % (max_ep_len*average_episode) == 0:
            plt.figure()
            env.plot()
            plt.plot(percentage_users)
            plt.ylim(0, 100)
            plt.xlabel('Movement step')
            plt.ylabel('The percentage of satisfied users (%)')
            plt.title('The percentage of satisfied users on each step')
            plt.savefig("result_step.png")
            # plt.show()
            plt.close()

            ################ save 3 models ######################
            ppo_agent0.save('ppo_agent0.pth')
            ppo_agent1.save('ppo_agent1.pth')
            ppo_agent2.save('ppo_agent2.pth')

        i_episode += 1

    avg_rewards = []
    for i in range(0,len(rewards_per_episode)-average_episode,average_episode):
        a = np.mean(rewards_per_episode[i:i+average_episode])
        for j in range(average_episode):
            avg_rewards.append(a)

    plt.figure()
    plt.plot(rewards_per_episode, label = 'reward on each 1 episode')
    plt.plot(avg_rewards, label = 'average reward on each ' +str(average_episode)+ ' episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('The total reward each episode')
    plt.legend()
    plt.savefig("result_reward.png")
    # plt.show()
    plt.close()

    avg_user = []
    for i in range(0,len(percentage_user_episode)-average_episode,average_episode):
        a = np.mean(percentage_user_episode[i:i+average_episode])
        for j in range(average_episode):
            avg_user.append(a)

    plt.figure()
    plt.plot(percentage_user_episode, label = 'the percentage of users on each 1 episode')
    plt.plot(avg_user, label = 'the percentage of on each '+ str(average_episode)+ ' episode')
    plt.xlabel('Episode')
    plt.ylabel('The percentage of users')
    plt.title('The percentage of users each episode')
    plt.legend()
    plt.savefig("result_user.png")
    # plt.show()
    plt.close()

    # Create a DataFrame with two columns
    df = pd.DataFrame({
    'Average Reward': rewards_per_episode,
    'Average User Satisfaction (%)': percentage_user_episode
    })

    # Save to Excel
    df.to_excel('training_results_proposed_CNN.xlsx', index=False)
