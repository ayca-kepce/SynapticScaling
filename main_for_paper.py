from model_analysis import *
from plotting_functions import *

##### Initial set-up
# Current directory is defined
directory = os.getcwd()

# Directories for plotting are defined
dir_data = directory + r'\figures\data\\'
dir_plot = directory + r'\figures\\'



##### Plotting the figures

### Hebbian learning, the third factor in the three-factor Hebbian learning, and adaptive set-point are active in all figures
hebbian_flag, three_factor_flag, adaptive_set_point_flag= 1, 1, 1



### Plotting Figure 2
# Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
E_scaling_flag = 1
P_scaling_flag = 1
S_scaling_flag = 1
flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_full]

analyze_model(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure2\\',
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure2\\',
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure2\\',
              run_simulation=0, save_results=1, plot_results=1)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure2\\',
                                  run_simulation=0, save_results =1, plot_results=1)



### Plotting Figure 3
# K parameter is set to different values for the full model

analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure3\\', K=0,
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure3\\', K=0.5,
              run_simulation=0, save_results=1, plot_results=1)

plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure3\\', K=0,
                                  run_simulation=0, save_results =1, plot_results=1)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure3\\', K=0.5,
                                  run_simulation=0, save_results =1, plot_results=1)



### Plotting Figure 4
# All synatic scaling mechanisms are blocked
E_scaling_flag = 0
P_scaling_flag = 0
S_scaling_flag = 0
flags_no_scaling = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_no_scaling]

analyze_model(4, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure4\\',
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure4\\',
              run_simulation=0, save_results=1, plot_results=1)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure4\\',
                                  run_simulation=0, save_results =1, plot_results=1)



### Plotting Figure 5
# Turning off the flag of E-to-E scaling
E_scaling_flag = 0
P_scaling_flag = 1
S_scaling_flag = 1
flags_E_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

# Turning off the flag of PV-to-E scaling
E_scaling_flag = 1
P_scaling_flag = 0
S_scaling_flag = 1
flags_P_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

# Turning off the flag of SST-to-E scaling
E_scaling_flag = 1
P_scaling_flag = 1
S_scaling_flag = 0
flags_S_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_E_off, flags_P_off, flags_S_off]

analyze_model(4, flags_list[0:1], dir_data = dir_data, dir_plot = dir_plot + r'figure5\\',
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure5\\',
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure5\\',
              run_simulation=0, save_results=1, plot_results=1)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + r'figure5\\',
                                  run_simulation=0, save_results =1, plot_results=1)



### Plotting Figure 6
# Turning on only the flag of E-to-E scaling
E_scaling_flag = 1
P_scaling_flag = 0
S_scaling_flag = 0
flags_only_E_on = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

# Turning on only the flag of PV-to-E scaling
E_scaling_flag = 0
P_scaling_flag = 1
S_scaling_flag = 0
flags_only_P_on = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

# Turning on only the flag of SST-to-E scaling
E_scaling_flag = 0
P_scaling_flag = 0
S_scaling_flag = 1
flags_only_S_on = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_only_E_on, flags_only_P_on, flags_only_S_on]

analyze_model(4, flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + r'figure6\\',
              run_simulation=0, save_results=1, plot_results=1)
analyze_model(24, flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + r'figure6\\',
              run_simulation=0, save_results=1, plot_results=1)
plot_testing_at_regular_intervals(flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + r'figure6\\',
                                  run_simulation=0, save_results =1, plot_results=1)

# Plotting margins are different for only SST-to-E scaling on case, thus the related flag is set to True
analyze_model(4, flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + r'figure6\\',
              flag_only_S_on=True, run_simulation=0, save_results=1, plot_results=1)
analyze_model(24, flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + r'figure6\\',
              flag_only_S_on=True, run_simulation=0, save_results=1, plot_results=1)
plot_testing_at_regular_intervals(flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + r'figure6\\',
                                  flag_only_S_on=True, run_simulation=0, save_results =1, plot_results=1)