"""Main experiment: R replications of the hub with the warm-up T0 from the pilot experiment"""

import csv
import math
import os
import random
import statistics

from scipy import stats

R = 25 # number of replications
ERROR_PRECISION = 0.05  # 5%
T0 = 120.0
RUN_LENGTH = 100000.0 # (min)

# measures
MEASURES = ["redirected_s1_per_minute", "redirected_s2_per_minute", "redirected_s3_per_minute",
            "redirected_total_per_minute", "throughput_per_minute",
            "utilization_s1", "utilization_s2", "utilization_s3",
            "queue_s1", "queue_s2", "queue_s3",
            "wait_s1", "wait_s2", "wait_s3",
            "wait_hub", "time_in_hub"]


# This class implement the replication method simulation from the S3 Reading, Sec. 4.1-4.2, pp. 12-16, with the same logic as amrSimulation.py
class ReplicationMethodSimulation:
    def __init__(self, arrival_rate=0.75, buffer_capacity=3, sim_duration=RUN_LENGTH, seed=42, warmup=T0):
        self.mean_interarrival = 1 / arrival_rate
        self.max_capacity = 1 + buffer_capacity  # 1 in service + 3 in buffer = 4
        self.warmup = warmup
        self.run_length = sim_duration  # T_E: the time the statistics are collected over
        self.sim_duration = warmup + sim_duration  # the run ends at T0 + T_E

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

        # Statistical Counters (reset at the end of the warm-up)
        self.total_arrivals = 0

        self.redirected_s1 = 0
        self.redirected_s2 = 0
        self.redirected_s3 = 0

        self.s1_started = 0
        self.s2_started = 0
        self.s3_started = 0

        self.total_delay_s1 = 0.0
        self.total_delay_s2 = 0.0
        self.total_delay_s3 = 0.0

        # AMRs that completed service in the hub. Redirected AMRs are not included.
        self.hub_completed = 0
        self.total_delay_hub = 0.0
        self.total_time_in_hub = 0.0

        # Time-integrated accumulators for time-average queue length and utilization
        self.area_q1 = 0.0
        self.area_q2 = 0.0
        self.area_q3 = 0.0
        self.area_s1 = 0.0
        self.area_s2 = 0.0
        self.area_s3 = 0.0

        # Schedule first arrival and the end of the warm-up
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)
        self.time_next_depart_s1 = float('inf')
        self.time_next_depart_s2 = float('inf')
        self.time_next_depart_s3 = float('inf')
        self.time_end_warmup = self.warmup if self.warmup > 0.0 else float('inf')

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

        if self.time_end_warmup < min_time:
            min_time = self.time_end_warmup
            next_event = 'end_warmup'

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

    def end_warmup(self):
        """End of warm-up event: discards the statistics collected in [0, T0). The system state is kept."""
        self.time_end_warmup = float('inf')  # disable warmup event

        self.total_arrivals = 0

        self.redirected_s1 = 0
        self.redirected_s2 = 0
        self.redirected_s3 = 0

        self.s1_started = 0
        self.s2_started = 0
        self.s3_started = 0

        self.total_delay_s1 = 0.0
        self.total_delay_s2 = 0.0
        self.total_delay_s3 = 0.0

        self.hub_completed = 0
        self.total_delay_hub = 0.0
        self.total_time_in_hub = 0.0

        self.area_q1 = 0.0
        self.area_q2 = 0.0
        self.area_q3 = 0.0
        self.area_s1 = 0.0
        self.area_s2 = 0.0
        self.area_s3 = 0.0

    def join_buffer(self, buffer, amr):
        """Places an AMR in a buffer and notes when it started waiting."""
        amr['buffer_arrival'] = self.sim_time
        buffer.append(amr)

    def complete(self, amr):
        """An AMR completed its service in the hub and exits."""
        self.hub_completed += 1
        self.total_delay_hub += amr['total_delay']
        self.total_time_in_hub += self.sim_time - amr['hub_arrival']

# state transition methods
    def arrive_s1(self):
        """Arrive S1: Schedules the next arrival, then starts the AMR, buffers it, or redirects it."""
        self.total_arrivals += 1
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)

        if self.num_in_s1 >= self.max_capacity:
            # if S1 is full, redirect to the depot
            self.redirected_s1 += 1
            return

        amr = {'hub_arrival': self.sim_time, 'total_delay': 0.0}
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
            next_amr['total_delay'] += delay
            self.start_s1(next_amr)

        self.arrive_s2(amr)

    def arrive_s2(self, amr):
        """Arrive S2: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s2 >= self.max_capacity:
            # if S2 is full, redirect to the depot
            self.redirected_s2 += 1
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
            next_amr['total_delay'] += delay
            self.start_s2(next_amr)

        if is_heavy:
            self.arrive_s3(amr)
        else:
            self.complete(amr)

    def arrive_s3(self, amr):
        """Arrive S3: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s3 >= self.max_capacity:
            # if S3 is full, redirect to the depot
            self.redirected_s3 += 1
            return

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
        amr = self.amr_in_s3
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
            next_amr['total_delay'] += delay
            self.start_s3(next_amr)

        self.complete(amr)

    def run(self):
        """Main Simulation Loop."""
        self.initialize()

        while True:
            next_event = self.timing()

            if self.sim_time > self.sim_duration:
                # Final area accumulation up to sim_duration
                self.sim_time = self.sim_duration
                self.update_time_avg_stats()
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
            elif next_event == 'end_warmup':
                self.end_warmup()

    def results(self):
        """The measures of this replication over the data-collection phase T_E, in the order of MEASURES."""
        te = self.run_length
        per_minute = 1.0 / te
        total_redirected = self.redirected_s1 + self.redirected_s2 + self.redirected_s3
        return [self.redirected_s1 * per_minute,
                self.redirected_s2 * per_minute,
                self.redirected_s3 * per_minute,
                total_redirected * per_minute,
                self.hub_completed * per_minute,
                self.area_s1 / te, self.area_s2 / te, self.area_s3 / te,
                self.area_q1 / te, self.area_q2 / te, self.area_q3 / te,
                self.total_delay_s1 / self.s1_started,
                self.total_delay_s2 / self.s2_started,
                self.total_delay_s3 / self.s3_started,
                self.total_delay_hub / self.hub_completed,
                self.total_time_in_hub / self.hub_completed]


def summary(replications, column):
    """Sample average, sample variance and 95% CI half-width.""" 
    values = [row[column] for row in replications]
    n = len(values)
    mean = statistics.mean(values)
    variance = statistics.variance(values)
    # The formula used to calculate the 95% CI half-width is from the section 4.1 of the S3 reading.
    half = stats.t.ppf(0.975, n - 1) * math.sqrt(variance) / math.sqrt(n)
    return mean, variance, half


if __name__ == "__main__":
    # list of R different random seeds
    seeds = random.sample(range(1000, 2000), R)

    # Run R independent replications
    replications = []
    for r in range(R):
        sim = ReplicationMethodSimulation(seed=seeds[r])
        sim.run()
        replications.append(sim.results())

    output_path = "experiment_results"
    os.makedirs(output_path, exist_ok=True)

    # Save the raw data: one row per replication
    with open(os.path.join(output_path, "experiment_raw_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replication", "seed"] + MEASURES)
        for r, row in enumerate(replications):
            writer.writerow([r + 1, seeds[r]] + row)

    # print sample average, sample variance and 95% CI half-width per measure
    print(f"\nSummary over R = {R} replications (T0 = {T0:g} min, T_E = {RUN_LENGTH:g} min)")
    print(f"{'measure':<30}{'average':>12}{'variance':>12}{'half-width':>12}{'H/average':>11}"
          f"{'meets eps':>11}{'95% CI':>24}")
    for column, measure in enumerate(MEASURES):
        mean, variance, half = summary(replications, column)
        meets = "yes" if half <= ERROR_PRECISION * mean else "no"
        print(f"{measure:<30}{mean:>12.6f}{variance:>12.3e}{half:>12.6f}{half / mean:>10.1%}{meets:>11}"
              f"    [{mean - half:.6f}, {mean + half:.6f}]")
