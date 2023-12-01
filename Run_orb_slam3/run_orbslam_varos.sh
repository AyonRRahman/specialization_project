#!/bin/bash

# Initialize variables with default values
vi_flag=false
v_flag=false
dataset_name=""
algorithm=""
current_time=""

# Function to get the current time
get_current_time() {
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
}

# Parse command-line options and arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --VI)
            vi_flag=true
            algorithm="Monocular Visual-Inertial Slam"
            ;;
        --V)
            v_flag=true
            algorithm="Monocular Visual Slam"
            ;;
        -d|--dataset)
            shift
            dataset_name="$1"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Ensure only one of the flags is provided
if $vi_flag && $v_flag; then
    echo "Error: Both --VI and --V flags are provided. Choose only one."
    exit 1
fi

# Get the current time
get_current_time

# Perform tasks based on the provided flag
if $vi_flag; then

    echo "Running $algorithm on Dataset: $dataset_name"
    # Add your --VI flag commands here
    if [ -n "$dataset_name" ]; then
        cd /ORB_SLAM3/Examples/Monocular-Inertial/

        # echo "Merged path: /Datasets/EuRoC/$dataset_name"
        ./mono_inertial_euroc ../../Vocabulary/ORBvoc.txt /Datasets/EuRoC/EuRoC_vi.yaml /Datasets/EuRoC/$dataset_name /Datasets/EuRoC/time_stamp.txt run_"$algorithm"_"$dataset_name"_"$current_time"

        mv f_run_"$algorithm"_"$dataset_name"_"$current_time".txt /Datasets/Run_orbslam3/Results/
        mv kf_run_"$algorithm"_"$dataset_name"_"$current_time".txt /Datasets/Run_orbslam3/Results/
        
        cd /Datasets/Run_orbslam3
    else
        echo "Error: Dataset name is required for --VI flag."
        exit 1
    fi


elif $v_flag; then
    
    echo "Running $algorithm on Dataset: $dataset_name"
    # Add your --VI flag commands here
    if [ -n "$dataset_name" ]; then
        cd /ORB_SLAM3/Examples/Monocular/

        # echo "Merged path: /Datasets/EuRoC/$dataset_name"
        ./mono_euroc ../../Vocabulary/ORBvoc.txt /Datasets/EuRoC/EuRoC.yaml /Datasets/EuRoC/$dataset_name /Datasets/EuRoC/time_stamp.txt run_"$algorithm"_"$dataset_name"_"$current_time"
        
        mv f_run_"$algorithm"_"$dataset_name"_"$current_time".txt /Datasets/Run_orbslam3/Results/
        mv kf_run_"$algorithm"_"$dataset_name"_"$current_time".txt /Datasets/Run_orbslam3/Results/

        cd /Datasets/Run_orbslam3
    else
        echo "Error: Dataset name is required for --VI flag."
        exit 1
    fi


else
    echo "Error: Either --VI or --V flag must be provided."
    exit 1
fi

# Print information about the algorithm, dataset, and time
echo "Algorithm: $algorithm"
echo "Dataset: $dataset_name"
echo "Time: $current_time"
