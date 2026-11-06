"""
Validation: one-factor-at-a-time (OFAT) sensitivity analysis of the hub model
"""

import csv
import os
import random

from replicationMethodSimulation import ReplicationMethodSimulation, MEASURES, summary

R = 10  # replications per setting

# Arrival rates to compare
ARRIVAL_RATES = [40 / 60, 45 / 60, 50 / 60]

if __name__ == "__main__":
    # list of R different random seeds, the same for every arrival rate
    seeds = random.sample(range(2000, 3000), R)

    # Run R replications for every arrival rate
    results = {} 
    # run simulation for every arrival according to the OFAT method
    for arrival_rate in ARRIVAL_RATES:
        replications = []
        for seed in seeds:
            sim = ReplicationMethodSimulation(arrival_rate=arrival_rate, seed=seed)
            sim.run()
            replications.append(sim.results())
        results[arrival_rate] = replications

    # Save the raw data to csv file
    output_path = "validation_results"
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "ofat_raw_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arrivals_per_hour", "replication", "seed"] + MEASURES)
        for arrival_rate, replications in results.items():
            for r, row in enumerate(replications):
                writer.writerow([arrival_rate * 60, r + 1, seeds[r]] + row)

    # Print the average +- 95% CI half-width of each measure for every arrival rate
    print(f"OFAT: arrival rate (other inputs at baseline, R = {R}, same seeds)")
    print(f"{'measure':<30}" + "".join(f"{f'{rate * 60:.0f} per hour':>22}" for rate in ARRIVAL_RATES))

    # we print the following measures: redirected_total_per_minute, throughput_per_minute, utilization_s2, wait_s2, time_in_hub
    # since these measure are relevant and can be predicted what changes we will expect.

    for measure in ["redirected_total_per_minute", "throughput_per_minute",
         "utilization_s2", "wait_s2", "time_in_hub"]:
        column = MEASURES.index(measure)
        line = f"{measure:<30}"
        for arrival_rate in ARRIVAL_RATES:
            mean, variance, half = summary(results[arrival_rate], column)
            line += f"{f'{mean:.4f} +- {half:.4f}':>22}"
        print(line)
