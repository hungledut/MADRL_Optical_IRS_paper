import matplotlib.pyplot as plt
import pandas as pd

import torch


# import our customed uav-environment and policy
from environment import IRS_env

from PPO_model_CNN import PPO
from arguments import parse_args
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse_args()

has_continuous_action_space = False

max_ep_len = 500                 # max timesteps in one episode
# max_training_timesteps = int(9e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
action_std = None
################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.05              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

# initialize a PPO agent
ppo_agent0 = PPO(209, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent1 = PPO(209, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent2 = PPO(209, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

if __name__ == '__main__':

    ppo_agent0.load('weights_ppo/ppo_agent0.pth')
    ppo_agent1.load('weights_ppo/ppo_agent1.pth')
    ppo_agent2.load('weights_ppo/ppo_agent2.pth')
    percentage_users = []
    percentage_users_UAV0 = []
    percentage_users_UAV1 = []
    percentage_users_UAV2 = []
    backhaul_capacity_UAV0 = []
    backhaul_capacity_UAV1 = []
    backhaul_capacity_UAV2 = []

    env = IRS_env(L=args.L, lambda_=args.lambda_, h_UAV=args.h_UAV, h_HAP=args.h_HAP, zenith_angle=args.zenith_angle, 
                  theta_i=args.theta_i, wo=args.wo, a=args.a, users=args.users, uavs=args.uavs, size=args.size,
                  varphi_=args.varphi_, v0=args.v0, tau=args.tau, noise_power_FSO=args.noise_power_FSO, P_FSO=args.P_FSO, 
                  B_FSO=args.B_FSO, noise_power=args.noise_power, B_RF=args.B_RF, r_th=args.r_th, max_step= args.max_step,
                  grid_num=args.grid_num, Nc=args.Nc, Hcl=args.Hcl)

    ###### test model ##########
    O_UAV0, O_UAV1, O_UAV2, CLWC = env.reset()

    for step_i in range(0,max_ep_len):
        # start = time.time()
        action0 = ppo_agent0.select_action(O_UAV0,CLWC)
        action1 = ppo_agent1.select_action(O_UAV1,CLWC)
        action2 = ppo_agent2.select_action(O_UAV2,CLWC)
        O_UAV0, O_UAV1, O_UAV2, CLWC, S, N_UAV0, N_UAV1, N_UAV2, backhaul_capacity = env.step([action0,action1,action2])
        percentage_users.append(S*100/args.users)
        percentage_users_UAV0.append(N_UAV0*100/args.users)
        percentage_users_UAV1.append(N_UAV1*100/args.users)
        percentage_users_UAV2.append(N_UAV2*100/args.users)
        backhaul_capacity_UAV0.append(backhaul_capacity[0])
        backhaul_capacity_UAV1.append(backhaul_capacity[1])
        backhaul_capacity_UAV2.append(backhaul_capacity[2])
        # end = time.time()
        # print('computational time = ', end - start)
        if step_i == 150 or step_i == 250 or step_i == 499:
            env.animation(step_i)

            ######################################### USERS ##############################################
            # Set global font size
            plt.rcParams.update({'font.size': 18})
            plt.figure(figsize=(8, 6))
            plt.plot(percentage_users, label = 'By all UAVs', color= 'black')
            plt.plot(percentage_users_UAV0,label = 'By UAV1', color= 'red')
            plt.plot(percentage_users_UAV1,label = 'By UAV2', color= 'green')
            plt.plot(percentage_users_UAV2,label = 'By UAV3', color= 'blue')
            plt.xlim(0,max_ep_len)
            plt.ylim(0, 100)
            plt.grid(True)
            plt.xlabel('Time slot (s)')
            plt.ylabel('The percentage of supported users (%)')
            plt.legend(fontsize = 15)
            # plt.title('The percentage of users supported by UAVs on each step')
            plt.savefig("result_step.pdf", bbox_inches='tight')
            # plt.show()
            plt.close()
            ######################################### BACKHAUL CAPACITY ######################################
            plt.figure(figsize=(8, 6))
            plt.plot(backhaul_capacity_UAV0, label = 'Backhaul Capacity of UAV0', color= 'red')
            plt.plot(backhaul_capacity_UAV1, label = 'Backhaul Capacity of UAV1', color= 'green')
            plt.plot(backhaul_capacity_UAV2, label = 'Backhaul Capacity of UAV2', color= 'blue')
            # Create a DataFrame with two columns
            df = pd.DataFrame({
            'backhaul_capacity_UAV0': backhaul_capacity_UAV0,
            'backhaul_capacity_UAV1': backhaul_capacity_UAV1,
            'backhaul_capacity_UAV2': backhaul_capacity_UAV2
            })
             # Save to Excel
            df.to_excel('backhaul_capacity.xlsx', index=False)
            df = pd.DataFrame({
                'percentage_users': percentage_users
            })
            df.to_excel('percentage_users.xlsx', index=False)

            plt.xlim(0,max_ep_len)
            plt.grid(True)
            plt.xlabel('Time slot (s)')
            plt.ylabel('Backhaul capacity (1e9 bps)')
            # plt.title('The Backhaul capacity on each step')
            plt.legend()
            plt.savefig("result_backhaul.pdf", bbox_inches='tight')
            # plt.show()
            plt.close()

            env.plot_cloud()
    