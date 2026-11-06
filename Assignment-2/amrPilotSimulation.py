"""Pilot experiment for choosing the warm T0 variable"""

import csv
import math
import os
import random
import statistics

import matplotlib.pyplot as plt
from scipy import stats

R = 20              # number of replications
RUN_LENGTH = 3000.0  # minutes per replication
INTERVAL = 30.0     # length of one recording interval in minutes
FIRST_SEED = 1001   # seeds 1001..1020, so these runs are independent of the main experiment


class AmrPilotSimulation:
    def __init__(self, arrival_rate=0.75, buffer_capacity=3, sim_duration=RUN_LENGTH, seed=42):
        self.mean_interarrival = 1 / arrival_rate
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

        # Statistical Counters for the current interval (reset at every record event)
        self.s1_started = 0
        self.s2_started = 0
        self.s3_started = 0

        self.total_delay_s1 = 0.0
        self.total_delay_s2 = 0.0
        self.total_delay_s3 = 0.0

        # Time-integrated accumulators for time-average queue length and utilization
        self.area_q1 = 0.0
        self.area_q2 = 0.0
        self.area_q3 = 0.0
        self.area_s1 = 0.0
        self.area_s2 = 0.0
        self.area_s3 = 0.0

        # One row per interval: [start, util_s1, util_s2, util_s3, queue_s1, queue_s2, queue_s3,
        #                        wait_s1, wait_s2, wait_s3]
        self.intervals = []

        # Schedule first arrival and the first record event
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)
        self.time_next_depart_s1 = float('inf')
        self.time_next_depart_s2 = float('inf')
        self.time_next_depart_s3 = float('inf')
        self.time_next_record = INTERVAL

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

        # Queue lengths (AMRs in buffer = max(0, total - 1))
        q1 = max(0, self.num_in_s1 - 1)
        q2 = max(0, self.num_in_s2 - 1)
        q3 = max(0, self.num_in_s3 - 1)

        # Server statuses (1 if busy, 0 if idle)
        s1 = 1 if self.num_in_s1 > 0 else 0
        s2 = 1 if self.num_in_s2 > 0 else 0
        s3 = 1 if self.num_in_s3 > 0 else 0

        self.area_q1 += q1 * dt
        self.area_q2 += q2 * dt
        self.area_q3 += q3 * dt
        self.area_s1 += s1 * dt
        self.area_s2 += s2 * dt
        self.area_s3 += s3 * dt

    def record(self):
        """Record event: saves the averages of the interval that just ended and starts a new one."""
        start = self.sim_time - INTERVAL
        utilization = [self.area_s1 / INTERVAL, self.area_s2 / INTERVAL, self.area_s3 / INTERVAL]
        queue = [self.area_q1 / INTERVAL, self.area_q2 / INTERVAL, self.area_q3 / INTERVAL]
        # Average wait in the buffer, or None if no AMR started service in this interval
        wait = [self.total_delay_s1 / self.s1_started if self.s1_started > 0 else None,
                self.total_delay_s2 / self.s2_started if self.s2_started > 0 else None,
                self.total_delay_s3 / self.s3_started if self.s3_started > 0 else None]
        self.intervals.append([start] + utilization + queue + wait)

        # Start a new interval
        self.s1_started = 0
        self.s2_started = 0
        self.s3_started = 0

        self.total_delay_s1 = 0.0
        self.total_delay_s2 = 0.0
        self.total_delay_s3 = 0.0

        # Time-integrated accumulators for time-average queue length and utilization
        self.area_q1 = 0.0
        self.area_q2 = 0.0
        self.area_q3 = 0.0
        self.area_s1 = 0.0
        self.area_s2 = 0.0
        self.area_s3 = 0.0

        self.time_next_record = self.sim_time + INTERVAL

    def join_buffer(self, buffer, amr):
        """Places an AMR in a buffer and notes when it started waiting."""
        amr['buffer_arrival'] = self.sim_time
        buffer.append(amr)

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
        self.s1_started += 1
        self.time_next_depart_s1 = self.sim_time + 1.0

    def depart_s1(self):
        """Depart S1: Starts the next AMR in the buffer and sends the finished AMR to S2."""
        amr = self.amr_in_s1
        self.num_in_s1 -= 1

        if self.num_in_s1 == 0:
            # if the buffer is empty, S1 becomes idle
            self.amr_in_s1 = None
            self.time_next_depart_s1 = float('inf')
        else:
            # else, pop the next AMR from the buffer and add its delay
            next_amr = self.q_s1.pop(0)
            delay = self.sim_time - next_amr['buffer_arrival']
            self.total_delay_s1 += delay
            self.start_s1(next_amr)

        self.arrive_s2(amr)

    def arrive_s2(self, amr):
        """Arrive S2: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s2 >= self.max_capacity:
            return

        self.num_in_s2 += 1
        if self.num_in_s2 == 1:
            self.start_s2(amr)
        else:
            self.join_buffer(self.q_s2, amr)

    def start_s2(self, amr):
        """Start S2: Decides the service type and gets its service time."""
        self.amr_in_s2 = amr
        self.s2_started += 1

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
            # if the buffer is empty, S2 becomes idle
            self.amr_in_s2 = None
            self.time_next_depart_s2 = float('inf')
        else:
            # else, pop the next AMR from the buffer and add its delay
            next_amr = self.q_s2.pop(0)
            delay = self.sim_time - next_amr['buffer_arrival']
            self.total_delay_s2 += delay
            self.start_s2(next_amr)

        if is_heavy:
            self.arrive_s3(amr)

    def arrive_s3(self, amr):
        """Arrive S3: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s3 >= self.max_capacity:
            return  # no need to count redirect for pilot

        self.num_in_s3 += 1
        if self.num_in_s3 == 1:
            self.start_s3(amr)
        else:
            self.join_buffer(self.q_s3, amr)

    def start_s3(self, amr):
        """Start S3: Heavy drive repair takes Uniform(6, 10) minutes."""
        self.amr_in_s3 = amr
        self.s3_started += 1
        self.time_next_depart_s3 = self.sim_time + self.uniform(6.0, 10.0)

    def depart_s3(self):
        """Depart S3: Starts the next AMR in the buffer; the finished AMR exits the hub."""
        self.num_in_s3 -= 1

        if self.num_in_s3 == 0:
            # if the buffer is empty, S3 becomes idle
            self.amr_in_s3 = None
            self.time_next_depart_s3 = float('inf')
        else:
            # else, pop the next AMR from the buffer and add its delay
            next_amr = self.q_s3.pop(0)
            delay = self.sim_time - next_amr['buffer_arrival']
            self.total_delay_s3 += delay
            self.start_s3(next_amr)

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


def ensemble_average(replications, column):
    """Ensemble average and 95% CI of one column for every interval, over all
    replications (S3 Reading, Sec. 5.1, pp. 22-24). Returns (averages, lows, highs)."""
    averages, lows, highs = [], [], []
    for k in range(len(replications[0])):
        values = [intervals[k][column] for intervals in replications
                  if intervals[k][column] is not None]
        mean = statistics.mean(values)
        half = stats.t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
        averages.append(mean)
        lows.append(mean - half)
        highs.append(mean + half)
    return averages, lows, highs


if __name__ == "__main__":
    replications = []
    for r in range(R):
        sim = AmrPilotSimulation(seed=FIRST_SEED + r)
        sim.run()
        replications.append(sim.intervals)

    # Save the raw data: one row per replication and interval
    output_path = "pilot_results"

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "pilot_raw_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replication", "seed", "interval_start",
                         "utilization_s1", "utilization_s2", "utilization_s3",
                         "queue_s1", "queue_s2", "queue_s3",
                         "wait_s1", "wait_s2", "wait_s3"])
        for r, intervals in enumerate(replications):
            for row in intervals:
                writer.writerow([r + 1, FIRST_SEED + r] + row)

    stations = ["S1", "S2", "S3"]
    measures = ["utilization", "number in queue (AMRs)", "wait in buffer (min)"]

    # calculate the ensemble average and 95% CI for every station and measure
    results = {}  # results[(station, measure)] = (averages, lows, highs)
    for i, station in enumerate(stations):
        for m, measure in enumerate(measures):
            column = 1 + 3 * m + i  # position of this station and measure in a row
            results[(station, measure)] = ensemble_average(replications, column)

    # plot one panel per station and measure
    starts = [row[0] for row in replications[0]]
    middles = [start + INTERVAL / 2 for start in starts]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for i, station in enumerate(stations):
        for m, measure in enumerate(measures):
            averages, lows, highs = results[(station, measure)]
            ax = axes[i][m]
            ax.fill_between(middles, lows, highs, alpha=0.25, label="95% CI")
            ax.plot(middles, averages, marker=".", label="ensemble average")
            ax.set_title(f"{station}: {measure}", fontsize=10)
            ax.grid(True, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("simulation time (min)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Average over {R} replications per {INTERVAL:g}-minute interval")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "pilot_mean_plot.png"), dpi=150)
