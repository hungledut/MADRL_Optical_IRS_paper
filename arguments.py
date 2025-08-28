import numpy as np
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # IRS parameters
    parser.add_argument("--L", type=str, default=0.4, help="IRS size (m)")
    parser.add_argument("--delta_IRS", type=str, default=0.9, help="IRS's reflection efficiency")
    parser.add_argument("--eta_UAV", type=str, default=0.9, help="Photodetector responsivity")
    parser.add_argument("--lambda_", type=list, default=[1550e-9,1555e-9,1560e-9], help="Optical wavelength of UAVs (m)")
    parser.add_argument("--h_UAV", type=str, default=350, help="UAV's altitude (m)")
    parser.add_argument("--h_HAP", type=str, default=20000, help="HAP's altitude (m)")
    parser.add_argument("--zenith_angle", type=str, default=60, help="Initial zenith angle (degree)")
    parser.add_argument("--theta_i", type=str, default=40, help="Incident angle (degree)")  
    parser.add_argument("--wo", type=str, default=2e-2, help="Waist of gaussian beam (m)")  
    parser.add_argument("--a", type=str, default=10e-2, help="Radius of lens (m)")  


    # UAV environment
    parser.add_argument("--users", type=str, default=200, help="The number of mobile users")  
    parser.add_argument("--uavs", type=str, default=3, help="The number of UAVs")  
    parser.add_argument("--size", type=str, default=2000, help="Target area size (m)")  
    parser.add_argument("--varphi_", type=str, default=np.pi/4, help="Half divergence angle of UAV (m)")  
    parser.add_argument("--v0", type=str, default=15, help="UAV's velocity (m)")  
    parser.add_argument("--tau", type=str, default=1, help="Time slot duration (s)")  

    # FSO parameters
    parser.add_argument("--noise_power_FSO", type=str, default=1e-5, help="FSO noise power (W)")
    parser.add_argument("--P_FSO", type=str, default=0.1, help="FSO power (W)")
    parser.add_argument("--B_FSO", type=str, default=1e9, help="FSO bandwidth (Hz)")

    # RF parameters
    parser.add_argument("--noise_power", type=str, default=1e-14, help="RF noise power (W)")
    parser.add_argument("--P_UAV", type=str, default=1e-14, help="UAV maximum power (W)")
    parser.add_argument("--B_RF", type=str, default=20e6, help="RF bandwidth (Hz)")
    parser.add_argument("--r_th", type=str, default=100e6, help="Target data rate threshold (bps)")

    # Other parameters
    parser.add_argument("--max_step", type=str, default=500, help="The number of steps")
    parser.add_argument("--grid_num", type=str, default=10, help="Grid size of user heatmap")

    # Cloud parameters
    parser.add_argument("--Nc", type=str, default=250, help="Droplet concentration (cm^-3)")
    parser.add_argument("--Hcl", type=str, default=2e3, help="Vertical extent of clouds (m)")

    # DRL parameters
    parser.add_argument("--episodes", type=str, default=4000, help="The number of episodes")

    return parser.parse_args()