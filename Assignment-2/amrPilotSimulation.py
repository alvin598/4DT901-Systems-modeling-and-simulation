import csv
import math
import os
import statistics

import matplotlib.pyplot as plt

from amrSimulation import AmrSimulation

R = 20           
RUN_LENGTH = 600.0 # 5 hours
BUCKET = 5.0 # time interval for averaging
FIRST_SEED = 1001   # seeds 1001..1020, so these runs are independent of the main experiment
OUT_DIR = "pilot_results"


class PilotSimulation(AmrSimulation):

    def initialize(self):
        super().initialize()
        n_buckets = int(RUN_LENGTH / BUCKET)
        # Area per bucket for: AMRs at S1, at S2, at S3, and in the whole hub
        self.bucket_area = [[0.0] * n_buckets for _ in range(4)]
        self.time_in_hub_sum = [0.0] * n_buckets   # time in hub of AMRs that completed in each bucket
        self.completed_count = [0] * n_buckets

    def update_time_avg_stats(self):
        # Spread the time since the last event over the buckets it covers.
        # This must happen before super(), which moves time_last_event forward.
        n = (self.num_in_s1, self.num_in_s2, self.num_in_s3,
             self.num_in_s1 + self.num_in_s2 + self.num_in_s3)
        t = self.time_last_event
        while t < self.sim_time and t < RUN_LENGTH:
            b = int(t // BUCKET)
            end = min(self.sim_time, (b + 1) * BUCKET)
            for k in range(4):
                self.bucket_area[k][b] += n[k] * (end - t)
            t = end
        super().update_time_avg_stats()

    
    def complete(self, amr):
        super().complete(amr)
        b = int(self.sim_time // BUCKET)
        if b < len(self.completed_count):
            self.time_in_hub_sum[b] += self.sim_time - amr['hub_arrival']
            self.completed_count[b] += 1

    # override to remove printing when running
    def report(self):
        pass

    def bucket_values(self):
        """One row per bucket: avg AMRs at S1, S2, S3, in hub, and avg time in hub."""
        rows = []
        for b in range(len(self.completed_count)):
            avg_time = (self.time_in_hub_sum[b] / self.completed_count[b]
                        if self.completed_count[b] > 0 else float('nan'))
            rows.append([self.bucket_area[k][b] / BUCKET for k in range(4)] + [avg_time])
        return rows


# 1. Run the replications
replications = []
for r in range(R):
    sim = PilotSimulation(sim_duration=RUN_LENGTH, seed=FIRST_SEED + r, warmup=0.0)
    sim.run()
    replications.append(sim.bucket_values())

# 2. Save the raw data
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "pilot_raw_data.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["replication", "seed", "bucket_start", "avg_in_s1", "avg_in_s2",
                     "avg_in_s3", "avg_in_hub", "avg_time_in_hub"])
    for r, rows in enumerate(replications):
        for b, values in enumerate(rows):
            writer.writerow([r + 1, FIRST_SEED + r, b * BUCKET] + values)

# 3. Ensemble average for every bucket (average across the replications), then plot
titles = ["AMRs at S1", "AMRs at S2", "AMRs at S3", "AMRs in the hub", "Time in hub (min)"]
times = [(b + 0.5) * BUCKET for b in range(int(RUN_LENGTH / BUCKET))]

fig, axes = plt.subplots(len(titles), 1, figsize=(9, 13), sharex=True)
for k, (ax, title) in enumerate(zip(axes, titles)):
    means = []
    for b in range(len(times)):
        values = [rows[b][k] for rows in replications if not math.isnan(rows[b][k])]
        means.append(statistics.mean(values))
    ax.plot(times, means)
    ax.set_title(f"{title}: ensemble average over {R} replications")
axes[-1].set_xlabel("simulation time (min)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pilot_mean_plot.png"), dpi=150)
print(f"Saved {OUT_DIR}/pilot_raw_data.csv and {OUT_DIR}/pilot_mean_plot.png")
