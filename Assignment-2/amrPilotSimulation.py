# -*- coding: utf-8 -*-
"""[4DV650] Assignment 2: Pilot experiment for choosing the warm-up length T0

The same model as amrSimulation.py, but instead of one average over the whole run it
records the averages of every station for each time interval of INTERVAL minutes.
The run starts empty and idle and has no warm-up. Averaging each interval over R
replications gives the ensemble averages that are plotted to choose T0
(S3 Reading, Sec. 5.1, Example 14).

"""

import csv
import math
import os
import random
import statistics

import matplotlib.pyplot as plt

R = 20              # number of replications
RUN_LENGTH = 300.0  # minutes per replication
INTERVAL = 10.0     # length of one recording interval in minutes
FIRST_SEED = 1001   # seeds 1001..1020, so these runs are independent of the main experiment
OUT_DIR = "pilot_results"


class AmrPilotSimulation:
    def __init__(self, arrival_rate=45.0, buffer_capacity=3, sim_duration=RUN_LENGTH, seed=42):
        """
        Pilot simulation of the Robotic Maintenance Hub.

        Parameters:
        - arrival_rate: Poisson arrival rate in AMRs per hour -> Mean interarrival = 60/45 minutes
        - buffer_capacity: Max buffer size = 3 (Total max capacity per station = 1 serving + 3 buffer = 4)
        - sim_duration: Simulation duration in minutes
        - seed: Random seed for reproducibility
        """
        self.mean_interarrival = 60.0 / arrival_rate
        self.max_capacity = 1 + buffer_capacity  # 1 in service + 3 in buffer = 4
        self.sim_duration = sim_duration

        random.seed(seed)

    def expon(self, mean):
        """Generates an Exponential random variate: X = -mean * ln(1 - U)"""
        return -mean * math.log(1.0 - random.random())

    def erlang(self, mean, k):
        """Generates an Erlang-k random variate with the given total mean: Sum of k Exponential variates."""
        return sum(self.expon(mean / k) for _ in range(k))

    def uniform(self, low, high):
        """Generates a Continuous Uniform random variate on (low, high)."""
        return low + (high - low) * random.random()

    def initialize(self):
        """Initialization Routine: Sets time, state, counters, and schedules the first events."""
        self.sim_time = 0.0
        self.time_last_event = 0.0

        # System State Variables: same as in amrSimulation.py
        self.num_in_s1 = 0
        self.q_s1 = []
        self.amr_in_s1 = None
        self.num_in_s2 = 0
        self.q_s2 = []
        self.amr_in_s2 = None
        self.num_in_s3 = 0
        self.q_s3 = []
        self.amr_in_s3 = None
        self.s2_is_heavy = False

        # Statistical counters for the current interval (reset at every record event)
        self.reset_interval_stats()

        # One row per interval: [start, util_s1, util_s2, util_s3, queue_s1, queue_s2, queue_s3,
        #                        wait_s1, wait_s2, wait_s3]
        self.intervals = []

        # Schedule first arrival and the first record event
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)
        self.time_next_depart_s1 = float('inf')
        self.time_next_depart_s2 = float('inf')
        self.time_next_depart_s3 = float('inf')
        self.time_next_record = INTERVAL

    def reset_interval_stats(self):
        """Zeroes the counters that are collected during one interval."""
        self.area_q = [0.0, 0.0, 0.0]       # area under number in queue, per station
        self.area_s = [0.0, 0.0, 0.0]       # area under server busy (0/1), per station
        self.total_delay = [0.0, 0.0, 0.0]  # total wait in buffer of AMRs that started service
        self.started = [0, 0, 0]            # number of AMRs that started service

    def timing(self):
        """Timing Routine: Scans FEL for the most imminent event."""
        min_time = self.time_next_depart_s3
        next_event = 'depart_s3'

        if self.time_next_depart_s2 < min_time:
            min_time = self.time_next_depart_s2
            next_event = 'depart_s2'

        if self.time_next_depart_s1 < min_time:
            min_time = self.time_next_depart_s1
            next_event = 'depart_s1'

        if self.time_next_arrival < min_time:
            min_time = self.time_next_arrival
            next_event = 'arrival'

        if self.time_next_record < min_time:
            min_time = self.time_next_record
            next_event = 'record'

        self.sim_time = min_time
        return next_event

    def update_time_avg_stats(self):
        """Updates continuous-time area accumulators."""
        dt = self.sim_time - self.time_last_event
        self.time_last_event = self.sim_time

        for i, n in enumerate((self.num_in_s1, self.num_in_s2, self.num_in_s3)):
            self.area_q[i] += max(0, n - 1) * dt    # AMRs in buffer
            self.area_s[i] += (1 if n > 0 else 0) * dt  # server busy or idle

    def record(self):
        """Record event: saves the averages of the interval that just ended and starts a new one."""
        start = self.sim_time - INTERVAL
        utilization = [area / INTERVAL for area in self.area_s]
        queue = [area / INTERVAL for area in self.area_q]
        # Average wait, or None if no AMR started service at that station in this interval
        wait = [self.total_delay[i] / self.started[i] if self.started[i] > 0 else None
                for i in range(3)]
        self.intervals.append([start] + utilization + queue + wait)

        self.reset_interval_stats()
        self.time_next_record = self.sim_time + INTERVAL

    def join_buffer(self, buffer, amr):
        """Places an AMR in a buffer and notes when it started waiting."""
        amr['buffer_arrival'] = self.sim_time
        buffer.append(amr)

    def pop_buffer(self, buffer, station):
        """Takes the next AMR from a buffer and counts how long it waited."""
        next_amr = buffer.pop(0)
        self.total_delay[station] += self.sim_time - next_amr['buffer_arrival']
        return next_amr

# state transition methods (same logic as amrSimulation.py)
    def arrive_s1(self):
        """Arrive S1: Schedules the next arrival, then starts the AMR, buffers it, or redirects it."""
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)

        if self.num_in_s1 >= self.max_capacity:
            return  # S1 full -> redirected to the depot

        amr = {}
        self.num_in_s1 += 1
        if self.num_in_s1 == 1:
            self.start_s1(amr)
        else:
            self.join_buffer(self.q_s1, amr)

    def start_s1(self, amr):
        """Start S1: Inspection and decontamination takes a constant 1 minute."""
        self.amr_in_s1 = amr
        self.started[0] += 1
        self.time_next_depart_s1 = self.sim_time + 1.0

    def depart_s1(self):
        """Depart S1: Starts the next AMR in the buffer and sends the finished AMR to S2."""
        amr = self.amr_in_s1
        self.num_in_s1 -= 1

        if self.num_in_s1 == 0:
            self.amr_in_s1 = None
            self.time_next_depart_s1 = float('inf')
        else:
            self.start_s1(self.pop_buffer(self.q_s1, 0))

        self.arrive_s2(amr)

    def arrive_s2(self, amr):
        """Arrive S2: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s2 >= self.max_capacity:
            return  # S2 full -> redirected to the depot

        self.num_in_s2 += 1
        if self.num_in_s2 == 1:
            self.start_s2(amr)
        else:
            self.join_buffer(self.q_s2, amr)

    def start_s2(self, amr):
        """Start S2: Decides the service type and gets its service time."""
        self.amr_in_s2 = amr
        self.started[1] += 1

        u = random.random()
        if u < 0.45:
            # Recalibration
            self.s2_is_heavy = False
            service_time = self.expon(2.0)
        elif u < 0.90:
            # Sensor Alignment
            self.s2_is_heavy = False
            service_time = self.erlang(4.0, 2)
        else:
            # Heavy Drive Repair diagnostic
            self.s2_is_heavy = True
            service_time = 1.0

        self.time_next_depart_s2 = self.sim_time + service_time

    def depart_s2(self):
        """Depart S2: Starts the next AMR in the buffer; the finished AMR exits or goes to S3."""
        amr = self.amr_in_s2
        is_heavy = self.s2_is_heavy  # Read now, as start_s2 overwrites it for the next AMR
        self.num_in_s2 -= 1

        if self.num_in_s2 == 0:
            self.amr_in_s2 = None
            self.time_next_depart_s2 = float('inf')
        else:
            self.start_s2(self.pop_buffer(self.q_s2, 1))

        if is_heavy:
            self.arrive_s3(amr)

    def arrive_s3(self, amr):
        """Arrive S3: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s3 >= self.max_capacity:
            return  # S3 full -> redirected to the depot

        self.num_in_s3 += 1
        if self.num_in_s3 == 1:
            self.start_s3(amr)
        else:
            self.join_buffer(self.q_s3, amr)

    def start_s3(self, amr):
        """Start S3: Heavy drive repair takes Uniform(6, 10) minutes."""
        self.amr_in_s3 = amr
        self.started[2] += 1
        self.time_next_depart_s3 = self.sim_time + self.uniform(6.0, 10.0)

    def depart_s3(self):
        """Depart S3: Starts the next AMR in the buffer; the finished AMR exits the hub."""
        self.num_in_s3 -= 1

        if self.num_in_s3 == 0:
            self.amr_in_s3 = None
            self.time_next_depart_s3 = float('inf')
        else:
            self.start_s3(self.pop_buffer(self.q_s3, 2))

    def run(self):
        """Main Simulation Loop."""
        self.initialize()

        while True:
            next_event = self.timing()

            if self.sim_time > self.sim_duration:
                break

            self.update_time_avg_stats()

            if next_event == 'arrival':
                self.arrive_s1()
            elif next_event == 'depart_s1':
                self.depart_s1()
            elif next_event == 'depart_s2':
                self.depart_s2()
            elif next_event == 'depart_s3':
                self.depart_s3()
            elif next_event == 'record':
                self.record()


if __name__ == "__main__":
    # 1. Run the replications, each with its own seed
    replications = []
    for r in range(R):
        sim = AmrPilotSimulation(seed=FIRST_SEED + r)
        sim.run()
        replications.append(sim.intervals)

    # 2. Save the raw data: one row per replication and interval
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "pilot_raw_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replication", "seed", "interval_start",
                         "utilization_s1", "utilization_s2", "utilization_s3",
                         "queue_s1", "queue_s2", "queue_s3",
                         "wait_s1", "wait_s2", "wait_s3"])
        for r, intervals in enumerate(replications):
            for row in intervals:
                writer.writerow([r + 1, FIRST_SEED + r] + row)

    # 3. Average every interval over the replications (ensemble average) and plot
    starts = [row[0] for row in replications[0]]
    middles = [start + INTERVAL / 2 for start in starts]
    measures = ["utilization", "number in queue (AMRs)", "wait in buffer (min)"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for i, station in enumerate(["S1", "S2", "S3"]):
        for m, measure in enumerate(measures):
            column = 1 + 3 * m + i  # position of this station and measure in a row
            averages = []
            for k in range(len(starts)):
                values = [intervals[k][column] for intervals in replications
                          if intervals[k][column] is not None]
                averages.append(statistics.mean(values) if values else None)
            ax = axes[i][m]
            ax.plot(middles, averages, marker=".")
            ax.set_title(f"{station}: {measure}", fontsize=10)
            ax.grid(True, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("simulation time (min)")
    fig.suptitle(f"Average over {R} replications per {INTERVAL:g}-minute interval")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pilot_mean_plot.png"), dpi=150)

    print(f"Saved {OUT_DIR}/pilot_raw_data.csv and {OUT_DIR}/pilot_mean_plot.png")
