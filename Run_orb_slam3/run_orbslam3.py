import subprocess
import time
import csv

def run_script_and_measure_time(script, options):
    # Record the start time
    start_time = time.time()
    terminate_flag = False
    try:
        # Redirect standard output and standard error to subprocess.PIPE
        process = subprocess.Popen([script] + options, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to complete
        process.communicate()
    except subprocess.CalledProcessError as e:
        # Handle errors here
        print(f"Error: {e}")
        return None
    except KeyboardInterrupt:
    # If KeyboardInterrupt (Ctrl+C) is received, terminate the process
        print('Terminated by user. Moving onto next dataset')
        process.terminate()
        terminate_flag=True
    
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    if terminate_flag:
        return -1
    
    return elapsed_time


vi_options = ["--VI","--V"]
csv_file = "Results/execution_times.csv"

datasets = ["MH01","MH02","MH03","MH04","MH05","MH06"]

# Specify the Bash script and options
bash_script = "./run_orbslam_varos.sh"

#running visual inertial for all the dataset
with open(csv_file, mode="w", newline="") as file:
    # Create a CSV writer object
    csv_writer = csv.writer(file)

    
    # Write the header row
    csv_writer.writerow(["Dataset", "--VI Flag", "Elapsed Time (s)"])

    # Loop through datasets and options
    for dataset in datasets:
        
        for option in vi_options:
            # Run the script and measure time
            print(f'running {option} for dataset {dataset}')
            
            elapsed_time = run_script_and_measure_time(bash_script, [option, "-d", dataset])

            # Write the data to the CSV file
            csv_writer.writerow([dataset, option, elapsed_time])

print(f"Execution times saved to {csv_file}")