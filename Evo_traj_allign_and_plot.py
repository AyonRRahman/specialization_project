#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from trajectory_plot import show_3d_plot_with_gt, show_views_together, show_3d_plot_same_figure,show_views_together_same_figure, plot2D_all, plot3D_all,plot2D_xy

from evo.core import trajectory, sync, metrics
from evo.tools import file_interface

def load_and_allign_traj(gt="/home/ayon/git/Thesis_data/trajectories/groundtruth.txt", est="/home/ayon/git/Thesis_data/trajectories/visual_inertial/CameraTrajectory_with_loop.txt",correct_scale=False):
    print("loading trajectories")
    traj_ref = file_interface.read_tum_trajectory_file(gt)
    traj_est = file_interface.read_tum_trajectory_file(est)
    print("registering and aligning trajectories")
    traj_ref_assoc, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    traj_est.align(traj_ref_assoc, correct_scale=correct_scale)
    
    return traj_est, traj_ref,traj_ref_assoc

def calculate_statistics(traj_ref, traj_est, print_stat=True):
    print("calculating APE")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()
    if print_stat:
        print(ape_statistics)

    print("calculating RPE")
    data = (traj_ref, traj_est)
    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part)
    rpe_metric.process_data(data)
    rpe_statistics = rpe_metric.get_all_statistics()

    if print_stat:
        print(rpe_statistics)
    
    return ape_statistics, rpe_statistics

def error_all_dataset(gt_direcory="/home/ayon/git/Thesis_data/trajectories/groundtruth.txt", estimation_dataset_dir='/home/ayon/git/orbslam3_docker/Datasets/Run_orbslam3/Results',data='Visual-Inertial Slam'):
    res = estimation_dataset_dir.split('/')[-1]
    # print(res)
    # return
    data='Visual Slam'
    files_list = sorted(os.listdir(estimation_dataset_dir))
    plot_dictionary = {}
    if data=='Visual-Inertial Slam':
        correct_scale = False
    else:
        correct_scale = True

    rpe_df = pd.DataFrame()
    ape_df = pd.DataFrame()

    for files in files_list:
        if files[-3:]=='csv':
            continue
        
        if 'kf' in files:
            continue
        
        if not data in files:
            continue

        name = files.split('_')[-2]
        traj_est, traj_ref, traj_ref_assoc = load_and_allign_traj(gt=gt_direcory, est=os.path.join(estimation_dataset_dir, files),correct_scale=correct_scale)
        ape_statistics, rpe_statistics = calculate_statistics(traj_ref_assoc, traj_est, print_stat=True)
        
        # print(traj_est._positions_xyz.shape[0])
        # print(ape_statistics, )
        ape_statistics['num_f'] = traj_est._positions_xyz.shape[0]
        rpe_statistics['num_f'] = traj_est._positions_xyz.shape[0]

    
        rpe_df = rpe_df.append(pd.Series(rpe_statistics, name=f'{name}'))
        ape_df = ape_df.append(pd.Series(ape_statistics, name=f'{name}'))

    # print(rpe_df)
    print(f'/home/ayon/git/Thesis_data/plots/rpe_{data}_{res}.csv')
    rpe_df.to_csv(f'/home/ayon/git/Thesis_data/plots/rpe_{data}_{res}.csv')
    ape_df.to_csv(f'/home/ayon/git/Thesis_data/plots/ape_{data}_{res}.csv')

        


def plot_all_dataset(gt_direcory="/home/ayon/git/Thesis_data/trajectories/groundtruth.txt", estimation_dataset_dir='/home/ayon/git/orbslam3_docker/Datasets/Run_orbslam3/Results1',data='Visual-Inertial Slam'):


    data='Visual Slam'
    files_list = sorted(os.listdir(estimation_dataset_dir))
    plot_dictionary = {}
    if data=='Visual-Inertial Slam':
        correct_scale = False
    else:
        correct_scale = True

    for files in files_list:
        if files[-3:]=='csv':
            continue
        
        if 'kf' in files:
            continue
        
        if not data in files:
            continue

        name = files.split('_')[-2]
        if  not (name == 'MH04' or name=='MH01'):

            continue

        traj_est, traj_ref, traj_ref_assoc = load_and_allign_traj(gt=gt_direcory, est=os.path.join(estimation_dataset_dir, files),correct_scale=correct_scale)
        xyz_est = traj_est._positions_xyz
        
        est = np.array([xyz_est[:,0],xyz_est[:,1], xyz_est[:,2]]).T
        print(name)
        print(est.shape)
        plot_dictionary[name] = est
    
    xyz_ref = traj_ref._positions_xyz
    gt = np.array([xyz_ref[:,0],xyz_ref[:,1],xyz_ref[:,2]]).T

    # plot = plot2D_xy(gt, plot_dictionary)
    # plot.savefig(f'/home/ayon/git/Thesis_data/plots/all_2d_plot_XY_{data}.jpg', format='jpg',dpi=300, bbox_inches='tight')

    plot = plot2D_all(gt, plot_dictionary)
    # plot.savefig(f'/home/ayon/git/Thesis_data/plots/all_2d_plot_{data}.jpg', format='jpg',dpi=300, bbox_inches='tight')

    plot = plot3D_all(gt, plot_dictionary)
    # plot.savefig(f'/home/ayon/git/Thesis_data/plots/all_3d_plot_{data}.jpg', format='jpg',dpi=300, bbox_inches='tight')
    
    plot.show()

if __name__=="__main__":
    # plot_all_dataset(estimation_dataset_dir='/home/ayon/git/Thesis_data/good_and_bad_results_for_plot/good')
    plot_all_dataset()
    # error_all_dataset(estimation_dataset_dir='/home/ayon/git/orbslam3_docker/Datasets/Run_orbslam3/Results')


    




# print("loading plot modules")
# from evo.tools import plot
# import matplotlib.pyplot as plt

# print("plotting")
# plot_collection = plot.PlotCollection("Example")
# # metric values
# fig_1 = plt.figure(figsize=(8, 8))
# plot.error_array(fig_1.gca(), ape_metric.error, statistics=ape_statistics,
#                  name="APE", title=str(ape_metric))
# plot_collection.add_figure("raw", fig_1)

# # trajectory colormapped with error
# fig_2 = plt.figure(figsize=(8, 8))
# plot_mode = plot.PlotMode.xy
# ax = plot.prepare_axis(fig_2, plot_mode)
# plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
# plot.traj_colormap(ax, traj_est, ape_metric.error, plot_mode,
#                    min_map=ape_statistics["min"],
#                    max_map=ape_statistics["max"],
#                    title="APE mapped onto trajectory")
# plot_collection.add_figure("traj (error)", fig_2)

# # trajectory colormapped with speed
# fig_3 = plt.figure(figsize=(8, 8))
# plot_mode = plot.PlotMode.xy
# ax = plot.prepare_axis(fig_3, plot_mode)
# speeds = [
#     trajectory.calc_speed(traj_est.positions_xyz[i],
#                           traj_est.positions_xyz[i + 1],
#                           traj_est.timestamps[i], traj_est.timestamps[i + 1])
#     for i in range(len(traj_est.positions_xyz) - 1)
# ]
# speeds.append(0)
# plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
# plot.traj_colormap(ax, traj_est, speeds, plot_mode, min_map=min(speeds),
#                    max_map=max(speeds), title="speed mapped onto trajectory")
# fig_3.axes.append(ax)
# plot_collection.add_figure("traj (speed)", fig_3)

# plot_collection.show()
