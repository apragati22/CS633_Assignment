import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

# ─── Argument Parser ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot times from run output files.")
parser.add_argument("folder_path", type=str, help="Path to the folder containing output files")
args = parser.parse_args()
folder_path = args.folder_path.rstrip("/")  # Remove trailing slash if present

# ─── Parameters ────────────────────────────────────────────────────────────
num_processes = [8, 16, 32, 64]
data_ids = ["64_64_64_3", "64_64_96_7"]
runs = [1, 2, 3, 4]

# ─── Data Collection ───────────────────────────────────────────────────────
records = []

for data_id in data_ids:
    for np in num_processes:
        for run in runs:
            filename = f"{folder_path}/output_{data_id}_{np}_run{run}.txt"
            try:
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 3:
                        times = list(map(float, lines[2].split(',')))
                        record = {
                            'data_id': data_id,
                            'num_processes': np,
                            'run': run,
                            'read_time': times[0],
                            'main_code_time': times[1],
                            'total_time': times[2]
                        }
                        records.append(record)
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

# ─── Plotting ──────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
time_types = ['read_time', 'main_code_time', 'total_time']

for data_id in data_ids:
    df_subset = df[df['data_id'] == data_id]

    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Iterate over the time types and plot them on the same graph
    for time_type, color, label in zip(time_types, ['blue', 'green', 'red'], ['Read Time', 'Main Code Time', 'Total Time']):
        stats = df_subset.groupby('num_processes')[time_type].agg(['mean', 'std']).reset_index()

        # Plot the mean values with error bars
        sns.lineplot(
            x='num_processes',
            y='mean',
            data=stats,
            marker='o',
            color=color,
            label=f'{label} - Mean'
        )
        plt.errorbar(
            stats['num_processes'],
            stats['mean'],
            yerr=stats['std'],
            fmt='o',
            color=color,
            capsize=5,
            label=f'{label} - Std Dev'
        )
    
    plt.title(f"Time vs Processes ({data_id})")
    plt.xlabel("Number of Processes")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{folder_path}/times_{data_id}_combined.png")
    plt.close()  # Close the figure after saving
