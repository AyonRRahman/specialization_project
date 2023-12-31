import pandas as pd 
import os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

# Set the figure size globally
plt.rcParams["figure.figsize"] = 12.8, 9.6


def plot3D_all(ground_truth, estimated_trajectories):
    """
    Plot ground truth and estimated trajectories in 3D.

    Parameters:
        ground_truth (numpy.ndarray): Array of shape (n, 3) representing ground truth trajectory.
        estimated_trajectories (dict): Dictionary where keys are names and values are arrays of shape (m, 3)
                                       representing estimated trajectories.

    Returns:
        None (displays the plot).

    Example:
        ground_truth = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        estimated_trajectories = {'Estimation 1': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                                   'Estimation 2': np.array([[0, 0, 0], [1, 0, -1], [2, 2, 0]])}

        plot3D_all(ground_truth, estimated_trajectories)
        plt.show()
    """
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth trajectory
    # ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth')
    estimated_trajectories['Ground Truth']=ground_truth
    # Define a colormap with distinct colors for each trajectory
    # colormap = plt.cm.get_cmap('viridis', len(estimated_trajectories) + 1)

    # Plot estimated trajectories
    for index, (name, trajectory) in enumerate(estimated_trajectories.items()):
        # color = colormap(index**2 / len(estimated_trajectories)**2)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=name)

        # Scatter plot for the start point
        if index == len(estimated_trajectories) - 1:
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], s=100, c='red', label='Start')
        else:
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], s=100, c='red')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Plot')
    ax.legend()

    # Display the plot
    return plt
    # plt.show()
# def plot3D_all(ground_truth, estimated_trajectories):
#     """
#     Plot ground truth and estimated trajectories in 3D.

#     Parameters:
#         ground_truth (numpy.ndarray): Array of shape (n, 3) representing ground truth trajectory.
#         estimated_trajectories (dict): Dictionary where keys are names and values are arrays of shape (m, 3)
#                                        representing estimated trajectories.

#     Returns:
#         None (displays the plot).

#     Example:
#         ground_truth = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
#         estimated_trajectories = {'Estimation 1': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
#                                    'Estimation 2': np.array([[0, 0, 0], [1, 0, -1], [2, 2, 0]])}

#         plot3D_all(ground_truth, estimated_trajectories)
#         plt.show()
#     """
#     # Create 3D plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot ground truth trajectory
#     ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth')

#     # Plot estimated trajectories
#     for index,(name, trajectory) in enumerate(estimated_trajectories.items()):
#         ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=name)

#         # Scatter plot for the start point
#         if index==len(estimated_trajectories)-1:
#             ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], s=100, c='red', label='Start')
#         else:
#             ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], s=100, c='red')

#     # Set labels and legend
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Trajectory Plot')
#     ax.legend()

#     # Display the plot
#     return plt
#     # plt.show()


def plot2D_xy(ground_truth, estimated_trajectories):
    """
    Plot ground truth and estimated trajectories in xy, yz, and zx views.

    Parameters:
        ground_truth (numpy.ndarray): Array of shape (n, 3) representing ground truth trajectory.
        estimated_trajectories (dict): Dictionary where keys are names and values are arrays of shape (m, 3)
                                       representing estimated trajectories.

    Returns:
        None (displays the plot).

    Example:
        ground_truth = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        estimated_trajectories = {'Estimation 1': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                                   'Estimation 2': np.array([[0, 0, 0], [1, 0, -1], [2, 2, 0]])}

        plot_trajectories(ground_truth, estimated_trajectories)
        plt.show()
    """
    # Create subplots
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    
    # Plot gt views
    # axs.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth')
    # axs[1].plot(ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth')
    # axs[2].plot(ground_truth[:, 2], ground_truth[:, 0], label='Ground Truth')
    estimated_trajectories['Ground Truth'] = ground_truth
    colormap = plt.cm.get_cmap('tab20', len(estimated_trajectories) + 1)
    
    #plot estimated views
    for index,(name, trajectory) in enumerate(estimated_trajectories.items()):
        color = colormap(index**2 / len(estimated_trajectories)**2)
        
        axs.plot(trajectory[:, 0], trajectory[:, 1], label=name, color=color)
        # axs[1].plot(trajectory[:, 1], trajectory[:, 2], label=name)
        # axs[2].plot(trajectory[:, 2], trajectory[:, 0], label=name)

        if index==len(estimated_trajectories)-1:
            axs.scatter(trajectory[0, 0], trajectory[0, 1],s=100,c='red', label='start')
            # axs[1].scatter(trajectory[0, 1], trajectory[0, 2],s=100,c='red', label='start')
            # axs[2].scatter(trajectory[0, 2], trajectory[0, 0],s=100,c='red', label='start')
        else:
            axs.scatter(trajectory[0, 0], trajectory[0, 1],s=100,c='red')
            # axs[1].scatter(trajectory[0, 1], trajectory[0, 2],s=100,c='red')
            # axs[2].scatter(trajectory[0, 2], trajectory[0, 0],s=100,c='red')



    axs.set_title('XY View')
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()

    
    # Set common labels
    
    # Display the plot
    return plt


def plot2D_all(ground_truth, estimated_trajectories):
    """
    Plot ground truth and estimated trajectories in xy, yz, and zx views.

    Parameters:
        ground_truth (numpy.ndarray): Array of shape (n, 3) representing ground truth trajectory.
        estimated_trajectories (dict): Dictionary where keys are names and values are arrays of shape (m, 3)
                                       representing estimated trajectories.

    Returns:
        None (displays the plot).

    Example:
        ground_truth = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        estimated_trajectories = {'Estimation 1': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                                   'Estimation 2': np.array([[0, 0, 0], [1, 0, -1], [2, 2, 0]])}

        plot_trajectories(ground_truth, estimated_trajectories)
        plt.show()
    """
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot gt views
    # axs[0].plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth')
    # axs[1].plot(ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth')
    # axs[2].plot(ground_truth[:, 2], ground_truth[:, 0], label='Ground Truth')
    estimated_trajectories['Ground Truth'] = ground_truth
    #plot estimated views
    for index,(name, trajectory) in enumerate(estimated_trajectories.items()):
        axs[0].plot(trajectory[:, 0], trajectory[:, 1], label=name)
        axs[1].plot(trajectory[:, 1], trajectory[:, 2], label=name)
        axs[2].plot(trajectory[:, 2], trajectory[:, 0], label=name)

        if index==len(estimated_trajectories)-1:
            axs[0].scatter(trajectory[0, 0], trajectory[0, 1],s=100,c='red', label='start')
            axs[1].scatter(trajectory[0, 1], trajectory[0, 2],s=100,c='red', label='start')
            axs[2].scatter(trajectory[0, 2], trajectory[0, 0],s=100,c='red', label='start')
        else:
            axs[0].scatter(trajectory[0, 0], trajectory[0, 1],s=100,c='red')
            axs[1].scatter(trajectory[0, 1], trajectory[0, 2],s=100,c='red')
            axs[2].scatter(trajectory[0, 2], trajectory[0, 0],s=100,c='red')



    axs[0].set_title('XY View')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()

    # Plot yz view
    # for name, trajectory in estimated_trajectories.items():
    axs[1].set_title('YZ View')
    axs[1].set_xlabel('Y')
    axs[1].set_ylabel('Z')
    axs[1].legend()

    # Plot zx view
    # for name, trajectory in estimated_trajectories.items():
    axs[2].set_title('ZX View')
    axs[2].set_xlabel('Z')
    axs[2].set_ylabel('X')
    axs[2].legend()

    # Set common labels
    
    # Display the plot
    return plt

# # Example usage:
# ground_truth = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
# estimated_trajectories = {'Estimation 1': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
#                            'Estimation 2': np.array([[0, 0, 0], [1, 0, -1], [2, 2, 0]])}

# plot_trajectories(ground_truth, estimated_trajectories)


def show_3d_plot_same_figure(trajectory, ground_truth):
    """
    Display a 3D plot with the estimated trajectory and ground truth trajectory side by side.

    Parameters:
    - trajectory (list): List of 3 elements representing X, Y, and Z coordinates of the estimated trajectory.
    - ground_truth (list): List of 3 elements representing X, Y, and Z coordinates of the ground truth trajectory.

    Returns:
    - plt.Figure: The Matplotlib Figure object containing the subplots.

    The function creates two subplots, each showing a 3D plot of the trajectory in red and ground truth in green:
    - Subplot 1: Estimated Trajectory
    - Subplot 2: Ground Truth Trajectory
    """

    # Unpack the data
    x, y, z = trajectory
    gt_x, gt_y, gt_z = ground_truth

    # Create a figure with 2 subplots
    fig = plt.figure()

    # Subplot 1
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot3D(x, y, z, 'red', label='Estimated Trajectory')  # Plot the estimated trajectory in red
    
    ax1.plot3D(gt_x, gt_y, gt_z, 'green', label='Ground Truth')  # Plot the ground truth trajectory in green
    sc2 = ax1.scatter3D(gt_x[0], gt_y[0], gt_z[0], c='b')  # Scatter plot for the ground truth trajectory
    sc1 = ax1.scatter3D(x[0], y[0], z[0], c='b', label='start')  # Scatter plot for the estimated trajectory
    sc1 = ax1.scatter3D(x[-1], y[-1], z[-1], c='orange', label='end')  # Scatter plot for the estimated trajectory
    

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Estimated Trajectory')
    ax1.legend()
    # Set labels and titles for Subplot 2
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')
    # ax2.set_title('Ground Truth Trajectory')

    # Adjust layout for better visualization
    plt.tight_layout()

    return plt

def show_3d_plot_with_gt(trajectory, ground_truth):
    """
    Display a 3D plot with the estimated trajectory and ground truth trajectory side by side.

    Parameters:
    - trajectory (list): List of 3 elements representing X, Y, and Z coordinates of the estimated trajectory.
    - ground_truth (list): List of 3 elements representing X, Y, and Z coordinates of the ground truth trajectory.

    Returns:
    - plt.Figure: The Matplotlib Figure object containing the subplots.

    The function creates two subplots, each showing a 3D plot of the trajectory in red and ground truth in green:
    - Subplot 1: Estimated Trajectory
    - Subplot 2: Ground Truth Trajectory
    """

    # Unpack the data
    x, y, z = trajectory
    gt_x, gt_y, gt_z = ground_truth

    # Create a figure with 2 subplots
    fig = plt.figure()

    # Subplot 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot3D(x, y, z, 'red', label='Estimated')  # Plot the estimated trajectory in red
    sc1 = ax1.scatter3D(x, y, z, c='red')  # Scatter plot for the estimated trajectory
    
    
    # Subplot 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot3D(gt_x, gt_y, gt_z, 'green',label='Ground truth')  # Plot the ground truth trajectory in green
    sc2 = ax2.scatter3D(gt_x, gt_y, gt_z, c='green')  # Scatter plot for the ground truth trajectory

    # Set labels and titles for Subplot 1
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Estimated Trajectory')
    ax1.legend()
    # Set labels and titles for Subplot 2
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Ground Truth Trajectory')
    ax2.legend()
    # Adjust layout for better visualization
    plt.tight_layout()

    return plt

def show_views_together(trajectory, ground_truth):
    """
    Display multiple 2D views of a trajectory and its ground truth.

    Parameters:
    - trajectory (list): List of 3 elements representing X, Y, and Z coordinates of the trajectory.
    - ground_truth (list): List of 3 elements representing X, Y, and Z coordinates of the ground truth.

    Returns:
    - plt.Figure: The Matplotlib Figure object containing the subplots.

    Each view is displayed in a 2x3 subplot arrangement:
    - XY view of the trajectory and ground truth.
    - Ground Truth XY view with start point in green and end point in red.
    - YZ view of the trajectory and ground truth.
    - Ground Truth YZ view with start point in green and end point in red.
    - ZX view of the trajectory and ground truth.
    - Ground Truth ZX view with start point in green and end point in red.
    """

    # Unpack the data
    x, y, z = trajectory
    gt_x, gt_y, gt_z = ground_truth

    # Create a 2x3 subplot figure with specified size
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Original XY view
    axs[0, 0].scatter(x, y, c='k', marker='o')  # Scatter plot for trajectory
    axs[0, 0].scatter(x[0], y[0], c='g', marker='o', s=100, label='Start point')  # Start point in green
    axs[0, 0].scatter(x[-1], y[-1], c='r', marker='o', s=100,label='End point')  # End point in red

    axs[0, 0].set_title('XY View')  # Set subplot title
    axs[0, 0].set_xlabel('X Label')  # Set X-axis label
    axs[0, 0].set_ylabel('Y Label')  # Set Y-axis label
    axs[0, 0].legend(loc=1)
    # Ground Truth XY view
    axs[1, 0].scatter(gt_x, gt_y, c='k', marker='o')  # Scatter plot for ground truth trajectory
    axs[1, 0].scatter(gt_x[0], gt_y[0], c='g', marker='o', s=100,label='Start point')  # Start point in green
    axs[1, 0].scatter(gt_x[-1], gt_y[-1], c='r', marker='o', s=100,label='End point')  # End point in red

    axs[1, 0].set_title('Ground Truth XY View')  # Set subplot title
    axs[1, 0].set_xlabel('X')  # Set X-axis label
    axs[1, 0].set_ylabel('Y')  # Set Y-axis label
    axs[1, 0].legend(loc=1)
    # Other views (you can customize the number of views as needed)

    # Original YZ view
    axs[0, 1].scatter(y, z, c='k', marker='o')
    axs[0, 1].scatter(y[0], z[0], c='g', marker='o', s=100,label='Start point')
    axs[0, 1].scatter(y[-1], z[-1], c='r', marker='o', s=100,label='End point')
    axs[0, 1].set_title('YZ View')
    axs[0, 1].set_xlabel('Y')
    axs[0, 1].set_ylabel('Z')
    axs[0, 1].legend(loc=1)
    # Ground Truth YZ view
    axs[1, 1].scatter(gt_y, gt_z, c='k', marker='o')
    axs[1, 1].scatter(gt_y[0], gt_z[0], c='g', marker='o', s=100,label='Start point')
    axs[1, 1].scatter(gt_y[-1], gt_z[-1], c='r', marker='o', s=100,label='End point')
    axs[1, 1].set_title('Ground Truth YZ View')
    axs[1, 1].set_xlabel('Y')
    axs[1, 1].set_ylabel('Z')
    axs[1, 1].legend(loc=1)
    # Original ZX view
    axs[0, 2].scatter(z, x, c='k', marker='o')
    axs[0, 2].scatter(z[0], x[0], c='g', marker='o', s=100,label='Start point')
    axs[0, 2].scatter(z[-1], x[-1], c='r', marker='o', s=100,label='End point')
    axs[0, 2].set_title('ZX View')
    axs[0, 2].set_xlabel('Z')
    axs[0, 2].set_ylabel('X')
    axs[0, 2].legend(loc=1)
    # Ground Truth ZX view
    axs[1, 2].scatter(gt_z, gt_x, c='k', marker='o')
    axs[1, 2].scatter(gt_z[0], gt_x[0], c='g', marker='o', s=100, label='Start point')
    axs[1, 2].scatter(gt_z[-1], gt_x[-1], c='r', marker='o', s=100, label='End point')
    axs[1, 2].set_title('Ground Truth ZX View')
    axs[1, 2].set_xlabel('Z')
    axs[1, 2].set_ylabel('X')
    axs[1,2].legend(loc=1)
    plt.tight_layout()
    
    return plt

def show_views_together_same_figure(trajectory, ground_truth):
    """
    Display multiple 2D views of a trajectory and its ground truth.

    Parameters:
    - trajectory (list): List of 3 elements representing X, Y, and Z coordinates of the trajectory.
    - ground_truth (list): List of 3 elements representing X, Y, and Z coordinates of the ground truth.

    Returns:
    - plt.Figure: The Matplotlib Figure object containing the subplots.

    Each view is displayed in a 2x3 subplot arrangement:
    - XY view of the trajectory and ground truth.
    - Ground Truth XY view with start point in green and end point in red.
    - YZ view of the trajectory and ground truth.
    - Ground Truth YZ view with start point in green and end point in red.
    - ZX view of the trajectory and ground truth.
    - Ground Truth ZX view with start point in green and end point in red.
    """

    # Unpack the data
    x, y, z = trajectory
    gt_x, gt_y, gt_z = ground_truth

    # Create a 2x3 subplot figure with specified size
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))

    # Original XY view
    axs[0].plot(x, y, c='red', label='Estimated')  # Scatter plot for trajectory
    
    axs[0].set_title('XY View')  # Set subplot title
    axs[0].set_xlabel('X')  # Set X-axis label
    axs[0].set_ylabel('Y')  # Set Y-axis label
    
    # Ground Truth XY view
    axs[0].plot(gt_x, gt_y, c='green', label = 'Ground_truth')  # Scatter plot for ground truth trajectory

    axs[0].scatter(gt_x[0], gt_y[0], c='b', marker='o', s=100)  # Start point in green
    axs[0].scatter(gt_x[-1], gt_y[-1], c='orange', marker='o', s=20)  # End point in red

    axs[0].scatter(x[0], y[0], c='b', marker='o', s=100, label='Start point')  # Start point in green
    axs[0].scatter(x[-1], y[-1], c='orange', marker='o', s=20,label='End point')  # End point in red

    axs[0].legend(loc=1)
    # Other views (you can customize the number of views as needed)

    # Original YZ view
    axs[1].plot(y, z, c='red',label='Estimated')
    axs[1].set_title('YZ View')
    axs[1].set_xlabel('Y')
    axs[1].set_ylabel('Z')
    # Ground Truth YZ view
    axs[1].plot(gt_y, gt_z, c='green',label = 'Ground truth')


    axs[1].scatter(gt_y[0], gt_z[0], c='b', marker='o', s=100)
    axs[1].scatter(gt_y[-1], gt_z[-1], c='orange', marker='o', s=20)
    axs[1].scatter(y[0], z[0], c='b', marker='o', s=100,label='Start point')
    axs[1].scatter(y[-1], z[-1], c='orange', marker='o', s=20,label='End point')

    axs[1].set_title('Ground Truth YZ View')
    axs[1].set_xlabel('Y')
    axs[1].set_ylabel('Z')
    axs[1].legend(loc=1)
    # Original ZX view
    axs[2].plot(z, x, c='red', label='Estimated')
    axs[2].set_title('ZX View')
    axs[2].set_xlabel('Z')
    axs[2].set_ylabel('X')
    # Ground Truth ZX view
    axs[2].plot(gt_z, gt_x, c='g',  label = 'Ground truth')
    
    
    axs[2].scatter(z[0], x[0], c='b', marker='o', s=100,label='Start point')
    axs[2].scatter(z[-1], x[-1], c='orange', marker='o', s=20,label='End point')
    
    axs[2].scatter(gt_z[0], gt_x[0], c='b', marker='o', s=100)
    axs[2].scatter(gt_z[-1], gt_x[-1], c='orange', marker='o', s=20)
    # axs[2].set_title('Ground Truth ZX View')
    axs[2].set_xlabel('Z')
    axs[2].set_ylabel('X')
    axs[2].legend(loc=1)
    plt.tight_layout()
    
    return plt


if __name__=='__main__':
    kf_file = '/home/ayon/git/spec_project_code/trajectories/kf_varos_enlighten_gan_run3.txt'


    # kf_file = '/home/ayon/git/orbslam3_docker/ORB_SLAM3/Examples/Monocular/kf_varos_enlighten_gan_run3.txt'

    df = pd.read_csv(kf_file, delimiter=' ',header=None)
    x = df[1].to_numpy()
    y = df[2].to_numpy()
    z = df[3].to_numpy()
    y,z = -z,y
    # z = -z
    # y = -y
    # x = x - x[0]
    # y = y - y[0]
    # z = z - z[0]

    gt_file = '/home/ayon/Dataset/Varos/2021-08-17_SEQ1/vehicle0/ground_truth_vehicle0_poses/vehicle0_poses_euler.csv'
    gt_df = pd.read_csv(gt_file)
    
    sampled_df = gt_df.iloc[::30]

    # print(f'origin {gt_df.shape}, sampled {sampled_df.shape}, est {df.shape}')
    gt_x = sampled_df['s_x_vw'].to_numpy()
    gt_y = sampled_df['s_y_vw'].to_numpy()
    gt_z = sampled_df['s_z_vw'].to_numpy()

    # gt_x = gt_x - gt_x[0]
    # gt_y -=gt_y[0]
    # gt_z -= gt_z[0]

    trajectory = [-x,-y,z]
    ground_truth = [gt_x, gt_y, gt_z]
    
    plt = show_3d_plot_same_figure(trajectory, ground_truth)
    # plt = show_views_together(trajectory, ground_truth)
    # plt = show_views_together_same_figure(trajectory, ground_truth)
    plt.show()