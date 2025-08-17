## Multi-Agent Deep Reinforcement Learning for UAV Placement in Optical IRS-Aided Aerial Networks

### Our proposed MADRL framework for Optical IRS-Aided Aerial Networks
<p align="center">
  <img src="images/proposed_framework.png" alt="framework" width="400">
</p>

### Train our proposed CP-PPO MADRL model🤖
We can try different IRS sizes as follows
~~~
py train.py --L 0.4 
~~~
### Test model🌗
~~~
py test.py 
~~~

<p align="center">
  <img src="images/UAV_animation.gif" alt="framework" width="500">
</p>

### Code Usage 

~~~
usage: main.py [-h] [--L L] [--delta_IRS DELTA_IRS] [--eta_UAV ETA_UAV] [--lambda_ LAMBDA_] [--h_UAV H_UAV] [--h_HAP H_HAP] [--zenith_angle ZENITH_ANGLE]
               [--theta_i THETA_I] [--wo WO] [--a A] [--users USERS] [--uavs UAVS] [--size SIZE] [--varphi_ VARPHI_] [--v0 V0] [--tau TAU]
               [--noise_power_FSO NOISE_POWER_FSO] [--P_FSO P_FSO] [--B_FSO B_FSO] [--noise_power NOISE_POWER] [--P_UAV P_UAV] [--B_RF B_RF] [--r_th R_TH]
               [--max_step MAX_STEP] [--grid_num GRID_NUM] [--Nc NC] [--Hcl HCL] [--episodes EPISODES]

options:
  -h, --help            show this help message and exit
  --L L                 IRS size (m)
  --delta_IRS DELTA_IRS
                        IRS's reflection efficiency
  --eta_UAV ETA_UAV     Photodetector responsivity
  --lambda_ LAMBDA_     Optical wavelength of UAVs (m)
  --h_UAV H_UAV         UAV's altitude (m)
  --h_HAP H_HAP         HAP's altitude (m)
  --zenith_angle ZENITH_ANGLE
                        Initial zenith angle (degree)
  --theta_i THETA_I     Incident angle (degree)
  --wo WO               Waist of gaussian beam (m)
  --a A                 Radius of lens (m)
  --users USERS         The number of mobile users
  --uavs UAVS           The number of UAVs
  --size SIZE           Target area size (m)
  --varphi_ VARPHI_     Half divergence angle of UAV (m)
  --v0 V0               UAV's velocity (m)
  --tau TAU             Time slot duration (s)
  --noise_power_FSO NOISE_POWER_FSO
                        FSO noise power (W)
  --P_FSO P_FSO         FSO power (W)
  --B_FSO B_FSO         FSO bandwidth (Hz)
  --noise_power NOISE_POWER
                        RF noise power (W)
  --P_UAV P_UAV         UAV maximum power (W)
  --B_RF B_RF           RF bandwidth (Hz)
  --r_th R_TH           Target data rate threshold (bps)
  --max_step MAX_STEP   The number of steps
  --grid_num GRID_NUM   Grid size of user heatmap
  --Nc NC               Droplet concentration (cm^-3)
  --Hcl HCL             Vertical extent of clouds (km)
  --episodes EPISODES   The number of episodes
~~~