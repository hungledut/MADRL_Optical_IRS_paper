import numpy as np
import math
import gymnasium as gym
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from matplotlib import cm
from matplotlib.colors import ListedColormap

class IRS_env(gym.Env):
    def __init__(self,
        ###################### IRS parameters ###########################
        L = 0.4, # (m) 
        lambda_ = 1550e-9, # (m)
        # d1 = 4000, d2 = 4000, # (m)
        h_UAV = 350, # (m)
        ######## HAP ###########
        h_HAP = 20000, # (m) ~ 20km
        zenith_angle = 60, # (degree)
        theta_r = 40, # (degree)
        theta_i = 40, # (degree)
        ########################
        wo = 2e-2, # (m) waist of gaussian beam
        a = 10e-2, # (m) radius of lens
        ###################### UAV Environment #########################
        users = 300,
        uavs = 3,
        size = 2000,
        varphi_ = np.pi/4,
        v0 = 15, # UAV's velocity (m/s)
        tau = 1,
        ##### FSO ######
        noise_power_FSO = 1e-5, # (Hz)
        P_FSO = 0.1, # (W)
        B_FSO = 1e9, # (Hz)
        noise_power = 1e-14, # (W)
        P_UAV = 10, # (W) -> 
        B_RF = 20e6, # (Hz) 
        r_th = 100e6, # (bps) ~ 20Mbps
        max_step = 300,
        grid_num = 10,
        ###### Cloud ######
        Nc = 250, # (cm^-3)
        Hcl = 2, # (km)
        cloud_moving_step = 15, #
        ####### Train or Test ? #######
        test = False,
        ###### Gauss-Markov Mobility Model for User Moving #########
        user_moving_percent = 0.2 # percentage of user moving
        ):
        ############### IRS parameters ##############
        self.Lx = L
        self.Ly = L 
        self.lambda_ = lambda_
        self.d1 = 40000
        self.d2 = 0
        self.h_UAV = h_UAV
        self.wd1 = 0
        ###### HAP ##########
        self.h_HAP = h_HAP 
        self.zenith_angle = zenith_angle
        self.sin_i = math.sin(math.radians(theta_i))
        self.sin_r = math.sin(math.radians(theta_r))
        ######################
        self.wo = wo
        self.a = a
        self.zR = np.pi*self.wo**2/self.lambda_ # Rayleigh range
        self.k = 2*np.pi/self.lambda_ # wave number
        self.divergence_angle = self.lambda_/(np.pi*2*self.wo)  # radian
        self.regime = 'unknown'
        ############## UAV Environment ##################################
        self.users = users
        self.uavs = uavs
        self.irs =  self.Lx*self.Ly/self.uavs
        self.size = size
        self.varphi_ = varphi_
        self.UAV_coverage = self.h_UAV*math.tan(self.varphi_)
        self.v_0 = v0
        self.tau = tau
        self.P_FSO = P_FSO
        self.B_FSO = B_FSO
        self.C_FSO = np.zeros(self.uavs)
        self.noise_power_FSO = noise_power_FSO
        self.noise_power = noise_power
        self.P_UAV = np.zeros(self.uavs) + 10
        self.B_RF = B_RF 
        self.r_th = r_th
        #################### Path Loss of Access Links #########
        self.K = 50
        self.psi_L = 1
        self.psi_M_real = np.random.normal(0, 1/np.sqrt(2))
        self.psi_M_imag = np.random.normal(0, 1/np.sqrt(2))
        self.d = 1
        self.lambda_c = 3e8/np.array([5.5e9,6e9,6.5e9])
        self.alpha = 2.7
        ####################
        self.max_step = max_step
        self.uavs_location = np.zeros((2, self.uavs))
        # self.hap_location = np.array([[300],[400]])
        self.users_location = np.clip(np.random.normal(loc=1000, scale=240, size=(2, self.users)),0,1999)
        self.satisfied_users = np.zeros(self.users)
        self.rate_UAV = np.zeros(self.users) - 1000
        self.UAV0_behavior = np.zeros((2, self.max_step))
        self.UAV1_behavior = np.zeros((2, self.max_step))
        self.UAV2_behavior = np.zeros((2, self.max_step))
        self.step_ = 0
        self.grid_num = grid_num
        self.grid_size = self.size/self.grid_num
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        # Dictionary maps the abstract actions to the directions
        self.action_space = gym.spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([0, 0]),  # remain stationary
            1: np.array([0,self.v_0*self.tau]),  # up
            2: np.array([-self.v_0*self.tau, 0]),  # left
            3: np.array([0, -self.v_0*self.tau]),  # down
            4: np.array([self.v_0*self.tau, 0]),  # right
        }
        ###### Cloud attenuation ##########
        self.Nc = Nc 
        self.Hcl = Hcl
        self.heatmap_cloud = 0 #CLWC map
        self.CLWC = 0
        self.cloud_moving_step = cloud_moving_step # if =0.1 -> 1 step = 10 cloud step 
        self.grid_num_cloud = 50
        self.grid_size_cloud = self.size/self.grid_num_cloud
        self.cloud_range = int(1/self.cloud_moving_step * self.max_step)
        ##### Train or Test ??? ###########
        self.test = test
        ###### Gauss-Markov Mobility for User Moving ###############
        self.s_markov = np.zeros(self.users)
        self.d_markov = np.zeros(self.users)
        self.s_mean = 0.67
        self.d_mean = 90* (180/np.pi) # convert to radian
        self.al = 0.5 
        self.random_user_list = random.sample(list(range(self.users)), 20)
        self.user_moving_percent = user_moving_percent
        ######### the number of users within coverage ########
        self.N_UAV0 = 0
        self.N_UAV1 = 0
        self.N_UAV2 = 0
        self.N_UAV = np.array([self.N_UAV0,self.N_UAV1,self.N_UAV2])
        ######### the number of supported users of each UAV ####
        self.S_UAV0 = 0
        self.S_UAV1 = 0
        self.S_UAV2 = 0
        self.S_UAV = np.array([self.S_UAV0,self.S_UAV1,self.S_UAV2])
        ######## Heatmap of UAV ##############
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users_unsatisfied = np.zeros((self.grid_num,self.grid_num))
    def d1d2_cal(self, UAV_i):
        uav_dis = math.sqrt(self.uavs_location[0,UAV_i]**2+self.uavs_location[1,UAV_i]**2)
        d_2 = math.sqrt(self.h_HAP**2 + (self.h_HAP*math.tanh(self.zenith_angle)+uav_dis)**2)
        return self.d1,d_2
    def G1_regime(self, d_2):
        # self.d1, d_2 = self.d1d2_cal()
        #####
        gLS = (2*np.pi*self.wo**2)/(4*np.pi*self.d1**2)
        gPD = (np.pi*self.a**2)/(4*np.pi*d_2**2)
        G1 = 4*np.pi*(4*np.pi*self.irs**2*abs(self.sin_r)*abs(self.sin_i))*gLS*gPD/(self.lambda_**4)
        return G1
    def G2_regime(self, d_2):
        # self.d1, d_2 = self.d1d2_cal()
        #####
        gLS = (2*np.pi*self.wo**2)/(4*np.pi*self.d1**2)
        G2 = 4*np.pi*self.irs*abs(self.sin_i)*gLS/(self.lambda_**2)
        return G2
    def G3_regime(self, d_2):
        # self.d1, d_2 = self.d1d2_cal()
        #####
        self.wd1 = self.wo*(1+(self.d1/self.zR)**2)**(1/2) # beamwidth at d1
        Rd1 = self.d1*(1+(self.zR/self.d1)**2)# curvature of beam's wavefront at d1
        v1 = 2*d_2/(self.k*self.wd1**2)
        v2 = d_2/Rd1
        #########
        sin_all = (self.sin_i/self.sin_r)**2
        Weqx = (self.wd1*abs(self.sin_r)/abs(self.sin_i)) * ((v1*sin_all)**2 + (v2*sin_all+1)**2)**(1/2)
        Weqy = self.wd1*(v1**2 + (v2+1)**2)**(1/2)
        #########
        G3 = erf(math.sqrt(np.pi/2)*(self.a/Weqx)) * erf(math.sqrt(np.pi/2)*(self.a/Weqy))
        return G3
    def S_cal(self, UAV_i):
        self.d1, d_2 = self.d1d2_cal(UAV_i)
        #####
        self.wd1 = self.wo*(1+(self.d1/self.zR)**2)**(1/2) # beamwidth at d1
        #########
        S1 = (self.lambda_**2 * d_2**2)/(np.pi*self.a**2*abs(self.sin_r))
        S2 = np.pi*self.G3_regime(d_2)*self.wd1**2/(2*abs(self.sin_i))
        return S1,S2
    def users_markov(self,random_user_list):
        s_random = np.random.normal(0, 1)
        d_random = np.random.normal(0, 45)
        self.s_markov = self.al*self.s_markov + (1-self.al)*self.s_mean + math.sqrt(1-self.al**2)*s_random
        self.d_markov = self.al*self.d_markov + (1-self.al)*self.d_mean + math.sqrt(1-self.al**2)*d_random
        for i in random_user_list:
            self.users_location[0,i] += self.s_markov[i]*math.cos(self.d_markov[i])
            self.users_location[1,i] += self.s_markov[i]*math.sin(self.d_markov[i])
        self.users_location = np.clip(self.users_location,0,self.size)
    def geometric_loss(self, UAV_i):
        geo_loss = 0
        S1, S2 = self.S_cal(UAV_i)
        _ , d_2 = self.d1d2_cal(UAV_i)
        if self.irs < S1:
            geo_loss = self.G1_regime(d_2)
            self.regime = 'quadratic'
        elif self.irs >= S1 and self.irs <= S2:
            geo_loss = self.G2_regime(d_2)
            self.regime = 'linear'
        else:
            geo_loss = self.G3_regime(d_2)
            self.regime = 'saturate'
    
        self.wd1 = self.wo*(1+(self.d1/self.zR)**2)**(1/2) # beamwidth at d1
        if self.G3_regime(d_2) < 2*d_2**2 * self.wd1 * abs(self.sin_i) / ( self.d1**2 * self.a**2 * abs(self.sin_r) ):
            S3 = math.sqrt(self.G3_regime(d_2))*self.lambda_*d_2*self.wd1/(self.a*math.sqrt(2*self.sin_i*self.sin_r))
            if self.irs <= S3:
                geo_loss = self.G1_regime(d_2)
                self.regime = 'quadratic'
            else:
                geo_loss = self.G3_regime(d_2)
                self.regime = 'saturate'
        return geo_loss
    def cloud_attenuation(self, UAV_i):
        if self.step_ % self.cloud_moving_step == 0:
            cloud_step = self.cloud_moving_step
            self.CLWC = self.heatmap_cloud[:,int(self.step_//cloud_step):int(self.step_//cloud_step+self.grid_num_cloud)]
        cloud_gain = []
        x_axis = int(abs(self.size - self.uavs_location[1,UAV_i]-1)//self.grid_size_cloud)
        y_axis = int(abs(self.uavs_location[0,UAV_i]-1)//self.grid_size_cloud)
        CLWC_ = self.CLWC[x_axis,y_axis]
        V = 1.002/(self.Nc*CLWC_)**0.6473
        # print(CLWC_)
        epsilon = 0
        if V > 50:
            epsilon = 1.6
        elif 6 < V and V <= 50:
            epsilon = 1.3
        elif 1 < V and V <= 6:
            epsilon = 0.16*V + 0.34
        elif 0.5 < V and V <= 1:
            epsilon = V - 0.5
        elif V <= 0.5:
            elsilon = 0

        B_dB = (3.91/V) * (self.lambda_/550)**(-epsilon)
        B = B_dB/(10**4 * math.log10(math.e))
        sec_eR = 1/(math.cos(math.radians(self.zenith_angle)))
        cloud_gain = np.exp(-B*self.Hcl*sec_eR)
        return cloud_gain
    def users_inside_UAV_coverage(self,d_UAV):
        # UAV Coverage
        connect_tem = np.zeros((self.uavs,self.users))
        d_argmin = np.argmin(d_UAV, axis=0) # Calcute the closest pair of UAV and User
        for n in range(self.users):
            argmin = d_argmin[n] # UAV closest to User
            if d_UAV[argmin,n] <= self.UAV_coverage :   # if < 400 (m) -> connect , may be connected to multiple UAVs
                connect_tem[argmin,n] = 1
        return connect_tem
    def path_loss_UAV(self,d_UAV):
        gain_UAV = np.zeros((self.uavs,self.users))
        for n in range(self.users):
            for k in range(self.uavs):
                psi_UAV = math.sqrt((math.sqrt(self.K/(1+self.K))*self.psi_L + math.sqrt(1/(1+self.K))*self.psi_M_real)**2 + math.sqrt(1/(1+self.K))*self.psi_M_imag**2)
                theta = -20*math.log10(4*3.14*self.d/self.lambda_c[k]) # (dB)
                theta = 10**(theta/10)
                g_UAV = (abs(psi_UAV)**2)*theta*(math.sqrt((d_UAV[k,n])**2 + (self.h_UAV)**2)/self.d)**(-self.alpha)
                gain_UAV[k,n] = g_UAV
        return gain_UAV
    def distance_UAVs_users(self):
        d_UAV = np.zeros((self.uavs,self.users))
        for k in range(self.uavs):
            for n in range(self.users):
                d_UAV[k,n] = np.linalg.norm(self.users_location[:,n] - self.uavs_location[:,k])
        return d_UAV
    def step(self,actions):
        ########################### Reset each step ##################################################
        self.satisfied_users = np.zeros(self.users)
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        self.users_markov(self.random_user_list) # Gauss-Markov mobility model for User moving
        self.P_UAV = np.zeros(self.uavs) + 10
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users_unsatisfied = np.zeros((self.grid_num,self.grid_num))
        ############################## psi_M is complex number ###############################
        self.psi_M_real = np.random.normal(0, 1/np.sqrt(2))
        self.psi_M_imag = np.random.normal(0, 1/np.sqrt(2))
        ##############################################################################################
        
        ################################# UAVs take the actions ################################################
        for (action,k) in zip(actions,range(self.uavs)):
            self.uavs_location[:,k] += self._action_to_direction[action]

        self.uavs_location = np.clip(self.uavs_location, 0,self.size) # Restrict UAVs' coordinates

        ####################################### UAV behavior ##########################################################
        self.UAV0_behavior[:,self.step_] = self.uavs_location[:,0]
        self.UAV1_behavior[:,self.step_] = self.uavs_location[:,1]
        self.UAV2_behavior[:,self.step_] = self.uavs_location[:,2]
        self.step_ += 1

        ################################# Geometric Loss ##############################
        geo_loss = []
        for UAV_i in range(self.uavs):
            geo_loss.append(self.geometric_loss(UAV_i))
    
        ################################## Cloud Attenuation ######################################
        cloud_gain = []
        for UAV_i in range(self.uavs):
            cloud_gain.append(self.cloud_attenuation(UAV_i))

        ################################## FSO Capacity ###########################################

        for UAV_i in range(self.uavs):
            SNR = self.P_FSO*geo_loss[UAV_i]*cloud_gain[UAV_i]/self.noise_power_FSO
            self.C_FSO[UAV_i] = self.B_FSO*math.log2(1+SNR)
        ###########################################################################################

        ## Distance
        d_UAV = self.distance_UAVs_users()

        # UAV Coverage
        connect_tem = self.users_inside_UAV_coverage(d_UAV)
        ## Signal-to-noise ratio(SNR)
        gain_UAV = self.path_loss_UAV(d_UAV)

        gain_UAV = gain_UAV*connect_tem
        ## Calculate needed power
        P_total_needed_index, P_total_needed = self.power_allocation(gain_UAV)

        # Temp
        # JUST ONE UAV , IF MULTIPLE UAVS, please change this one
        for UAV_i in range(self.uavs):
            C_FSO_temp = self.C_FSO[UAV_i]
            for i in range(P_total_needed_index.shape[1]):
                if self.P_UAV[UAV_i] < P_total_needed[UAV_i,P_total_needed_index[UAV_i,i]]:
                    break
                if C_FSO_temp < self.r_th:
                    break
                # Allocate Data Rate and Power
                C_FSO_temp -= self.r_th
                self.P_UAV[UAV_i] -= P_total_needed[UAV_i,P_total_needed_index[UAV_i,i]]
                self.satisfied_users[P_total_needed_index[UAV_i,i]] = 1
        ##############################

        ##################### Heat map of Users ###########################
        users_list = []
        for i in range(self.users):
            users_list.append([self.users_location[0,i],self.users_location[1,i]])

        if len(users_list) > 0:
            for i in range(len(users_list)):
                x = int(abs(users_list[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list[i][1]-1)//self.grid_size)
                self.heatmap_users[x,y] += 1
        
        users_list_0 = []
        for i in range(self.users):
            if connect_tem[0,i] == 1:
                users_list_0.append([self.users_location[0,i],self.users_location[1,i]])
        if len(users_list_0) > 0:
            for i in range(len(users_list_0)):
                x = int(abs(users_list_0[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_0[i][1]-1)//self.grid_size)
                self.heatmap_UAV0[x,y] += 1

        users_list_1 = []
        for i in range(self.users):
            if connect_tem[1,i] == 1:
                users_list_1.append([self.users_location[0,i],self.users_location[1,i]])
        if len(users_list_1) > 0:
            for i in range(len(users_list_1)):
                x = int(abs(users_list_1[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_1[i][1]-1)//self.grid_size)
                self.heatmap_UAV1[x,y] += 1
        
        users_list_2 = []
        for i in range(self.users):
            if connect_tem[2,i] == 1:
                users_list_2.append([self.users_location[0,i],self.users_location[1,i]])

        if len(users_list_2) > 0:
            for i in range(len(users_list_2)):
                x = int(abs(users_list_2[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_2[i][1]-1)//self.grid_size)
                self.heatmap_UAV2[x,y] += 1

        users_list_satisfied = []
        for i in range(self.users):
            if self.satisfied_users[i] == 0:
                users_list_satisfied.append([self.users_location[0,i],self.users_location[1,i]])

        if len(users_list_satisfied) > 0:
            for i in range(len(users_list_satisfied)):
                x = int(abs(users_list_satisfied[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_satisfied[i][1]-1)//self.grid_size)
                self.heatmap_users_unsatisfied[x,y] += 1
        ##############################################################################

        State_0 = np.array([self.C_FSO[0]/(1e9),np.sum(self.satisfied_users),self.step_])
        State_1 = np.array([self.C_FSO[1]/(1e9),np.sum(self.satisfied_users),self.step_])
        State_2 = np.array([self.C_FSO[2]/(1e9),np.sum(self.satisfied_users),self.step_])
        
        O_UAV0 = np.concatenate((self.uavs_location[:,0],self.uavs_location[:,1],self.uavs_location[:,2],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV0,self.grid_num**2),State_0)) #,np.reshape(self.heatmap_CLWC(),10**2)
        O_UAV1 = np.concatenate((self.uavs_location[:,1],self.uavs_location[:,2],self.uavs_location[:,0],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV1,self.grid_num**2),State_1))
        O_UAV2 = np.concatenate((self.uavs_location[:,2],self.uavs_location[:,0],self.uavs_location[:,1],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV2,self.grid_num**2),State_2))

        S = np.sum(self.satisfied_users)
        self.N_UAV0 = np.sum(connect_tem[0,:])
        self.N_UAV1 = np.sum(connect_tem[1,:])
        self.N_UAV2 = np.sum(connect_tem[2,:])

        self.S_UAV0 = np.sum(connect_tem[0,:]*self.satisfied_users)
        self.S_UAV1 = np.sum(connect_tem[1,:]*self.satisfied_users)
        self.S_UAV2 = np.sum(connect_tem[2,:]*self.satisfied_users)

        return O_UAV0, O_UAV1, O_UAV2, self.CLWC, S, self.S_UAV0, self.S_UAV1, self.S_UAV2, self.C_FSO/(1e9)
    def power_allocation(self,gain_UAV):
        P_total_needed = np.zeros((self.uavs,self.users))
        for UAV_i in range(self.uavs):
             # the number of users supported by UAV_i = Backhaul Capacity/ r_th
            for i in range(gain_UAV.shape[1]):
                if gain_UAV[UAV_i,i] == 0: # means no connection to UAV
                    P_total_needed[UAV_i,i] = 10000 
                else:
                    P_total_needed[UAV_i,i] = 2**(self.r_th/(self.B_RF) - 1) * (self.noise_power/gain_UAV[UAV_i,i])
        P_total_needed_index = np.argsort(P_total_needed, axis=1)
        return P_total_needed_index, P_total_needed
    def plot(self):
        ######################################### UAVs' colour ################################
        color_uav = ['red','green','blue']
        ######################################### For Visualization ################################

        ## Distance
        d_UAV = self.distance_UAVs_users()

        plt.figure()
        ###################################### Drawing Cloud #########################################
        x = np.linspace(0+self.grid_size_cloud//2, self.size - self.grid_size_cloud//2, self.grid_num_cloud) 
        y = np.linspace(0+self.grid_size_cloud//2, self.size - self.grid_size_cloud//2, self.grid_num_cloud)
        fig, ax = plt.subplots()

        # Lấy 80% đầu của cmap 'Greys' để làm màu nhạt
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        greys_light = cm.get_cmap('Greys', 256)
        new_cmap = ListedColormap(greys_light(np.linspace(0, 0.7, 256)))
        # print(self.heatmap_CLWC())
        c = ax.pcolormesh(x, y, self.CLWC[::-1,:], cmap=new_cmap, shading='auto',vmin=0.0, vmax=np.max(self.heatmap_cloud))
        # Gắn colorbar bên cạnh
        cb = plt.colorbar(c, ax=ax)
        cb.set_label('CLWC')
        ################################## UAVs' Coverage #############################################

        # Lấy axes hiện tại
        ax = plt.gca()
        # Vẽ hình tròn quanh điểm
        for UAV_i in range(self.uavs):
            ax.add_patch(Circle((self.uavs_location[0,UAV_i],self.uavs_location[1,UAV_i]), self.UAV_coverage, fill=False, color = color_uav[UAV_i], linewidth=1))
            plt.gca().set_aspect('equal')
        
        # Cài đặt giới hạn hiển thị
        plt.xlim(-50, self.size + 50)
        plt.ylim(-50, self.size + 50)

        ######### BS  #############
        plt.scatter(0,0, label = 'Starting Position of UAV', marker = ',', color = 'yellow',linewidths=1,edgecolors='black',s=100)
        ######### HAP #############
        # plt.scatter(self.hap_location[0,0],self.hap_location[1,0], label = 'HAP with IRS', marker = '*', color = 'y',linewidths=1,edgecolors='black',s=200)

        # plt.scatter(self.users_location[0,:],self.users_location[1,:], marker = 'o', color = 'c')
        # plt.scatter(self.uavs_location[0,0],self.uavs_location[1,0], label = 'UAV0', marker = ',', color = 'r',linewidths=1,edgecolors='black')
        ######### UAVs ############
        for UAV_i in range(self.uavs):
            plt.scatter(self.uavs_location[0,UAV_i],self.uavs_location[1,UAV_i], label = 'UAV' + str(UAV_i), marker = ',', color = color_uav[UAV_i] ,linewidths=1,edgecolors='black')

        ######## Users ############
        connect_tem = self.users_inside_UAV_coverage(d_UAV)
        for user_i in range(self.users):
            if connect_tem[0,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = color_uav[0],s=10)
            elif connect_tem[0,user_i] == 1 and self.satisfied_users[user_i] == 1:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[0],s=10,linewidths=1,edgecolors='black')
            elif connect_tem[1,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = color_uav[1],s=10)
            elif connect_tem[1,user_i] == 1 and self.satisfied_users[user_i] == 1:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[1],s=10,edgecolors='black')
            elif connect_tem[2,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = color_uav[2],s=10)
            elif connect_tem[2,user_i] == 1 and self.satisfied_users[user_i] == 1:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[2],s=10,edgecolors='black')
            else:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = 'gray',s=10)

        plt.scatter(self.UAV0_behavior[0,:],self.UAV0_behavior[1,:], color = 'r', s=0.5)
        plt.scatter(self.UAV1_behavior[0,:],self.UAV1_behavior[1,:], color = 'g', s=0.5)
        plt.scatter(self.UAV2_behavior[0,:],self.UAV2_behavior[1,:], color = 'b', s=0.5)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        # plt.title('IRS-assisted FSO Communication in UAV Environment \n' + 'Step '+ str(self.step_))
        plt.legend()
        plt.savefig("UAV_moving.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        ######################################### For Table ################################
        #Dữ liệu cho bảng
        table_data = [
            ['Backhaul Capacity', str(round(self.C_FSO[0] / 1e9, 2)) + ' (Gbps)',str(round(self.C_FSO[1] / 1e9, 2)) + ' (Gbps)',str(round(self.C_FSO[2] / 1e9, 2)) + ' (Gbps)'],
            ['Threshold Rate', str(self.r_th // 1e6) + ' (Mbps)', str(self.r_th // 1e6) + ' (Mbps)', str(self.r_th // 1e6) + ' (Mbps)'],
            ['The number Supported Users', str(int(np.sum(self.S_UAV0))), str(int(np.sum(self.S_UAV1))), str(int(np.sum(self.S_UAV2)))],
            ['Users within coverage', str(int(np.sum(self.N_UAV0))), str(int(np.sum(self.N_UAV1))), str(int(np.sum(self.N_UAV2)))],
            ['UAV max power', '10 (W)','10 (W)','10 (W)'],
            ['UAV remaining power', str(round(self.P_UAV[0], 2)) + ' (W)', str(round(self.P_UAV[1], 2)) + ' (W)', str(round(self.P_UAV[2], 2)) + ' (W)']
        ]

        # Tạo figure và axis
        fig, ax = plt.subplots()
        ax.axis('off')  # Tắt trục

        # Tạo bảng với tiêu đề cột
        column_labels = ['Parameter', 'UAV0','UAV1','UAV2']
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels,
            cellLoc='center',
            loc='center'
        )

        # Tùy chỉnh font size
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)  # Tăng kích thước bảng

        # Tiêu đề
        plt.title('Step ' + str(self.step_), fontsize=14, pad=20)
        plt.savefig("parameters_overtime.png")
        plt.close()
        
    def reset(self):
        self.uavs_location = np.zeros((2, self.uavs))
        # np.random.seed(42)
        self.users_location = np.clip(np.random.normal(loc=1000, scale=240, size=(2, self.users)),0,1999)
        self.satisfied_users = np.zeros(self.users)
        self.UAV0_behavior = np.zeros((2, self.max_step))
        self.UAV1_behavior = np.zeros((2, self.max_step))
        self.UAV2_behavior = np.zeros((2, self.max_step))
        self.step_ = 0
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        self.P_UAV = np.zeros(self.uavs) + 10
        ############ Heatmap #############################
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users_unsatisfied = np.zeros((self.grid_num,self.grid_num))
        ######### the number of users within coverage ########
        self.N_UAV0 = 0
        self.N_UAV1 = 0
        self.N_UAV2 = 0
        self.N_UAV = np.array([self.N_UAV0,self.N_UAV1,self.N_UAV2])
        ######### the number of supported users of each UAV ####
        self.S_UAV0 = 0
        self.S_UAV1 = 0
        self.S_UAV2 = 0
        self.S_UAV = np.array([self.S_UAV0,self.S_UAV1,self.S_UAV2])
        ################### CLoud ######################################################
        self.heatmap_cloud = self.generate_cloud_matrix(self.grid_num_cloud, self.grid_num_cloud+self.cloud_range)#np.random.uniform(0.5, 15.5, (self.grid_num_cloud, self.grid_num_cloud+self.cloud_range))
        self.CLWC = self.heatmap_cloud[:,0:self.grid_num_cloud]

        ############################### Reset User Mobility ############################
        self.s_markov = np.zeros(self.users)
        self.d_markov = np.zeros(self.users)
        self.random_user_list = random.sample(list(range(self.users)), int(self.user_moving_percent*self.users))

        # S1, S2 = self.S_cal()
        ################################# Geometric Loss ##############################
        # geo_loss = self.geometric_loss()

        State_0 = np.array([self.C_FSO[0]/(1e9),np.sum(self.satisfied_users),self.step_])
        State_1 = np.array([self.C_FSO[1]/(1e9),np.sum(self.satisfied_users),self.step_])
        State_2 = np.array([self.C_FSO[2]/(1e9),np.sum(self.satisfied_users),self.step_])

        ####### Observations ######
        O_UAV0 = np.concatenate((self.uavs_location[:,0],self.uavs_location[:,1],self.uavs_location[:,2],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV0,self.grid_num**2),State_0)) #,np.reshape(self.heatmap_CLWC(),10**2)
        O_UAV1 = np.concatenate((self.uavs_location[:,1],self.uavs_location[:,2],self.uavs_location[:,0],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV1,self.grid_num**2),State_1))
        O_UAV2 = np.concatenate((self.uavs_location[:,2],self.uavs_location[:,0],self.uavs_location[:,1],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV2,self.grid_num**2),State_2))
        
        if self.test == True:
            return O_UAV0,O_UAV1,O_UAV2,self.CLWC
            
        return O_UAV0,O_UAV1,O_UAV2, self.CLWC

    def beam_vizualization(self):

        # Circle of Beamfootprint
        theta = np.linspace(0, 2 * np.pi, 100)
        r = self.wd1  # Beamwidth 
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # SQUARE of IRS
        half_L = self.Lx / 2
        square_x = [-half_L, half_L, half_L, -half_L, -half_L]
        square_y = [-half_L, -half_L, half_L, half_L, -half_L]

        # Vẽ hình tròn
        plt.figure(figsize=(6,6))
        plt.plot(x, y, label='Beamfootprint', color = 'red')
        plt.plot(square_x, square_y, 'b-', label='IRS')
        plt.gca().set_aspect('equal')  # Đảm bảo hình tròn không bị méo
        plt.title("Beamfootprint at IRS")
        plt.grid(True)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.savefig("beamfootprint_at_IRS.png")
    def generate_cloud_matrix(self,height, width, num_clouds=100, min_clwc=5, max_clwc=15):
        clwc_matrix = np.zeros((height, width))

        for _ in range(num_clouds):
        # Kích thước đám mây
            cloud_h = np.random.randint(5, 15)
            cloud_w = np.random.randint(5, 15)
        
        # Vị trí ngẫu nhiên cho đám mây
            top = np.random.randint(0, height - cloud_h)
            left = np.random.randint(0, width - cloud_w)
        
        # Tạo phân bố Gaussian cho CLWC trong đám mây
            y = np.linspace(-1, 1, cloud_h)
            x = np.linspace(-1, 1, cloud_w)
            xv, yv = np.meshgrid(x, y)
            gaussian = np.exp(-(xv**2 + yv**2) * 2)  # Hệ số 4 làm nó nhỏ gọn hơn

        # Scale về giá trị CLWC thực tế
            clwc = np.random.uniform(min_clwc, max_clwc) # range of each center of CLOUD 
            cloud_patch = clwc * gaussian

        # Chèn vào ma trận
            clwc_matrix[top:top+cloud_h, left:left+cloud_w] += cloud_patch

        clwc_matrix = np.clip(clwc_matrix,1,15)
        return clwc_matrix
    def plot_cloud(self):

        x = np.linspace(0+self.grid_size_cloud//2, self.size - self.grid_size_cloud//2, self.grid_num_cloud) 
        y = np.linspace(0+self.grid_size_cloud//2, self.size - self.grid_size_cloud//2, self.grid_num_cloud)
        fig, ax = plt.subplots()

        # Lấy 80% đầu của cmap 'Greys' để làm màu nhạt
        greys_light = cm.get_cmap('Greys', 256)
        new_cmap = ListedColormap(greys_light(np.linspace(0, 0.7, 256)))
        # print(self.heatmap_CLWC())
        c = ax.pcolormesh(x, y, self.CLWC[::-1,:], cmap=new_cmap, shading='auto',vmin=0.0, vmax=np.max(self.heatmap_cloud))
        # Gắn colorbar bên cạnh
        cb = plt.colorbar(c, ax=ax)
        cb.set_label('CLWC')
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.savefig("cloud_map.png")
        # plt.show()
        plt.close()

    def animation(self,step_i):
        ######################################### UAVs' colour ################################
        color_uav = ['red','green','blue']
        ######################################### For Visualization ################################

        ## Distance
        d_UAV = self.distance_UAVs_users()
        # Set global font size
        plt.rcParams.update({'font.size': 18})
        plt.figure()
        ###################################### Drawing Cloud #########################################
        x = np.linspace(0+self.grid_size_cloud//2, self.size - self.grid_size_cloud//2, self.grid_num_cloud) 
        y = np.linspace(0+self.grid_size_cloud//2, self.size - self.grid_size_cloud//2, self.grid_num_cloud)
        fig, ax = plt.subplots()

        # Lấy 80% đầu của cmap 'Greys' để làm màu nhạt
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        greys_light = cm.get_cmap('Greys', 256)
        new_cmap = ListedColormap(greys_light(np.linspace(0, 0.7, 256)))
        # print(self.heatmap_CLWC())
        c = ax.pcolormesh(x, y, self.CLWC[::-1,:], cmap=new_cmap, shading='auto',vmin=0.0, vmax=np.max(self.heatmap_cloud))
        # Gắn colorbar bên cạnh
        cb = plt.colorbar(c, ax=ax)
        cb.set_label('CLWC')
        ################################## UAVs' Coverage #############################################

        # Lấy axes hiện tại
        ax = plt.gca()
        # Vẽ hình tròn quanh điểm
        for UAV_i in range(self.uavs):
            ax.add_patch(Circle((self.uavs_location[0,UAV_i],self.uavs_location[1,UAV_i]), self.UAV_coverage, fill=False, color = color_uav[UAV_i], linewidth=1))
            plt.gca().set_aspect('equal')
        
        # Cài đặt giới hạn hiển thị
        plt.xlim(-50, self.size + 50)
        plt.ylim(-50, self.size + 50)

        ######### BS  #############
        plt.scatter(0,0, label = 'Starting Position', marker = ',', color = 'yellow',linewidths=1,edgecolors='black',s=100)
        ######### HAP #############
        # plt.scatter(self.hap_location[0,0],self.hap_location[1,0], label = 'HAP with IRS', marker = '*', color = 'y',linewidths=1,edgecolors='black',s=200)

        # plt.scatter(self.users_location[0,:],self.users_location[1,:], marker = 'o', color = 'c')
        # plt.scatter(self.uavs_location[0,0],self.uavs_location[1,0], label = 'UAV0', marker = ',', color = 'r',linewidths=1,edgecolors='black')
        ######### UAVs ############
        for UAV_i in range(self.uavs):
            plt.scatter(self.uavs_location[0,UAV_i],self.uavs_location[1,UAV_i], label = 'UAV' + str(UAV_i+1), marker = ',', color = color_uav[UAV_i] ,linewidths=1,edgecolors='black',s=80)

        ######## Users ############
        connect_tem = self.users_inside_UAV_coverage(d_UAV)
        for user_i in range(self.users):
            if connect_tem[0,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = color_uav[0],s=15)
            elif connect_tem[0,user_i] == 1 and self.satisfied_users[user_i] == 1:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[0],s=15,linewidths=1,edgecolors='black')
            elif connect_tem[1,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = color_uav[1],s=15)
            elif connect_tem[1,user_i] == 1 and self.satisfied_users[user_i] == 1:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[1],s=15,edgecolors='black')
            elif connect_tem[2,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = color_uav[2],s=15)
            elif connect_tem[2,user_i] == 1 and self.satisfied_users[user_i] == 1:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[2],s=15,edgecolors='black')
            else:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = '^', color = 'gray',s=15)

        plt.scatter(self.UAV0_behavior[0,:],self.UAV0_behavior[1,:], color = 'r', s=0.5)
        plt.scatter(self.UAV1_behavior[0,:],self.UAV1_behavior[1,:], color = 'g', s=0.5)
        plt.scatter(self.UAV2_behavior[0,:],self.UAV2_behavior[1,:], color = 'b', s=0.5)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        # plt.title('IRS-assisted FSO Communication in UAV Environment \n' + 'Step '+ str(self.step_))
        plt.legend(fontsize=12)
        plt.savefig("UAV_moving"+ str(step_i) +".pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        ######################################### For Table ################################
        #Dữ liệu cho bảng
        table_data = [
            ['Backhaul Capacity', str(round(self.C_FSO[0] / 1e9, 2)) + ' (Gbps)',str(round(self.C_FSO[1] / 1e9, 2)) + ' (Gbps)',str(round(self.C_FSO[2] / 1e9, 2)) + ' (Gbps)'],
            ['Threshold Rate', str(self.r_th // 1e6) + ' (Mbps)', str(self.r_th // 1e6) + ' (Mbps)', str(self.r_th // 1e6) + ' (Mbps)'],
            ['The number Supported Users', str(int(np.sum(self.S_UAV0))), str(int(np.sum(self.S_UAV1))), str(int(np.sum(self.S_UAV2)))],
            ['Users within coverage', str(int(np.sum(self.N_UAV0))), str(int(np.sum(self.N_UAV1))), str(int(np.sum(self.N_UAV2)))],
            ['UAV max power', '10 (W)','10 (W)','10 (W)'],
            ['UAV remaining power', str(round(self.P_UAV[0], 2)) + ' (W)', str(round(self.P_UAV[1], 2)) + ' (W)', str(round(self.P_UAV[2], 2)) + ' (W)']
        ]

        # Tạo figure và axis
        fig, ax = plt.subplots()
        ax.axis('off')  # Tắt trục

        # Tạo bảng với tiêu đề cột
        column_labels = ['Parameter', 'UAV0','UAV1','UAV2']
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels,
            cellLoc='center',
            loc='center'
        )

        # Tùy chỉnh font size
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)  # Tăng kích thước bảng

        # Tiêu đề
        plt.title('Step ' + str(self.step_), fontsize=14, pad=20)
        plt.savefig("parameters_overtime.png")
        plt.close()

    



    


        


