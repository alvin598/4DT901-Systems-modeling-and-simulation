""" 
Maintenance Hub simulation
The structure of the simulation is based on the example S2 [Code] Simulation_validation.py.
"""

import math
import random

class AmrSimulation:
    def __init__(self, arrival_rate=45.0, buffer_capacity=3, sim_duration=100000.0, seed=42):
        """
        Simulation of the Robotic Maintenance Hub.

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
        """Initialization Routine: Sets time, counters, and schedules 1st arrival."""
        self.sim_time = 0.0
        self.time_last_event = 0.0

        # System State Variables: AMRs at each station (serving + queue), the AMRs
        # in each buffer, and the AMR being served
        self.num_in_s1 = 0
        self.q_s1 = []
        self.amr_in_s1 = None
        self.num_in_s2 = 0
        self.q_s2 = []
        self.amr_in_s2 = None
        self.num_in_s3 = 0
        self.q_s3 = []
        self.amr_in_s3 = None
        self.s2_is_heavy = False  # Service type of the AMR being served in S2

        # Statistical Counters
        self.total_arrivals = 0
        self.s1_arrivals = 0
        self.s2_arrivals = 0
        self.s3_arrivals = 0

        self.redirected_s1 = 0
        self.redirected_s2 = 0
        self.redirected_s3 = 0

        self.s1_started = 0
        self.s2_started = 0
        self.s3_started = 0

        self.s1_completed = 0
        self.s2_completed = 0
        self.s3_completed = 0

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

        # Schedule first arrival
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)
        self.time_next_depart_s1 = float('inf')
        self.time_next_depart_s2 = float('inf')
        self.time_next_depart_s3 = float('inf')

    def timing(self):
        """Timing Routine: Scans FEL for the most imminent event. """
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

    def join_buffer(self, buffer, amr):
        """Places an AMR in a buffer and notes when it started waiting."""
        amr['buffer_arrival'] = self.sim_time
        buffer.append(amr)

    def complete(self, amr):
        """An AMR completed its service in the hub and exits. the total time in the hub is calculated"""
        self.hub_completed += 1
        self.total_delay_hub += amr['total_delay']
        self.total_time_in_hub += self.sim_time - amr['hub_arrival']

# state transition methods
    def arrive_s1(self):
        """first arrival at S1 increments the total."""
        self.total_arrivals += 1

        # Schedule next arrival
        self.time_next_arrival = self.sim_time + self.expon(self.mean_interarrival)

        if self.num_in_s1 >= self.max_capacity:
            # if s1 is full, redirect to the depot
            self.redirected_s1 += 1
            return

        amr = {'hub_arrival': self.sim_time, 'buffer_arrival': self.sim_time, 'total_delay': 0.0}
        self.s1_arrivals += 1
        self.num_in_s1 += 1
        if self.num_in_s1 == 1:
            # if s1 is idle, start service immediately
            self.start_s1(amr)
        else:
            # if s1 is busy but buffer available, join the buffer
            self.join_buffer(self.q_s1, amr)

    def start_s1(self, amr):
        """Start S1: Inspection and decontamination takes a constant 1 minute."""
        self.amr_in_s1 = amr
        self.s1_started += 1
        self.time_next_depart_s1 = self.sim_time + 1.0

    def depart_s1(self):
        """Depart S1: Pops the S1 buffer and sends the finished AMR to S2."""
        amr = self.amr_in_s1
        self.s1_completed += 1
        self.num_in_s1 -= 1

        if self.num_in_s1 == 0:
            # if the buffer is empty the se becomes idle
            self.amr_in_s1 = None
            self.time_next_depart_s1 = float('inf')
        else:
            # else, the pop from buffer and update with the delay
            next_amr = self.q_s1.pop(0)
            delay = self.sim_time - next_amr['buffer_arrival']
            self.total_delay_s1 += delay
            next_amr['total_delay'] += delay
            self.start_s1(next_amr)

        self.arrive_s2(amr)

    def arrive_s2(self, amr):
        """Arrive S2: Starts the AMR, places it in the buffer, or redirects it."""
        if self.num_in_s2 >= self.max_capacity:
            # if s2 is full, redirect to the depot
            self.redirected_s2 += 1
            return

        self.s2_arrivals += 1
        self.num_in_s2 += 1
        if self.num_in_s2 == 1:
            # if s2 is idle, start service immediately
            self.start_s2(amr)
        else:
            # if s2 is busy but buffer available, join the buffer
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
        """Depart S2: Pops the S2 buffer, and the finished AMR exits or goes to S3."""
        amr = self.amr_in_s2
        is_heavy = self.s2_is_heavy  # Read now, as start_s2 overwrites it for the next AMR
        self.s2_completed += 1
        self.num_in_s2 -= 1

        if self.num_in_s2 == 0:
            # if the buffer is empty, S2 becomes idle
            self.amr_in_s2 = None
            self.time_next_depart_s2 = float('inf')
        else:
            # else, the pop from buffer and update with the delay
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
            # if s3 is full, redirect to the depot
            self.redirected_s3 += 1
            return

        self.s3_arrivals += 1
        self.num_in_s3 += 1
        if self.num_in_s3 == 1:
            # if s3 is idle, start service immediately
            self.start_s3(amr)
        else:
            # if s3 is busy but buffer available, join the buffer
            self.join_buffer(self.q_s3, amr)

    def start_s3(self, amr):
        """Start S3: Heavy drive repair takes Uniform(6, 10) minutes."""
        self.amr_in_s3 = amr
        self.s3_started += 1
        self.time_next_depart_s3 = self.sim_time + self.uniform(6.0, 10.0)

    def depart_s3(self):
        """Depart S3: Pops the S3 buffer and the finished AMR exits the hub."""
        amr = self.amr_in_s3
        self.s3_completed += 1
        self.num_in_s3 -= 1

        if self.num_in_s3 == 0:
            # if the buffer is empty, S3 becomes idle
            self.amr_in_s3 = None
            self.time_next_depart_s3 = float('inf')
        else:
            # else, the pop from buffer and update with the delay
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

        self.report()

    def report(self):
        """Generates performance report."""
        per_hour = 60.0 / self.sim_time
        total_redirected = self.redirected_s1 + self.redirected_s2 + self.redirected_s3

        print("=" * 70)
        print("SIMULATION REPORT: ROBOTIC MAINTENANCE HUB")
        print("=" * 70)
        print(f"Simulation Duration:              {self.sim_duration:.1f} min ({self.sim_duration/60:.1f} hours)")
        print(f"Total AMRs Arrived:               {self.total_arrivals}")
        print(f"AMRs Completed in Hub:            {self.hub_completed}")
        print(f"Throughput:                       {self.hub_completed * per_hour:.4f} AMRs/hour")
        print("-" * 70)
        print("REDIRECTED TO THE DEPOT:")
        print(f"  - At S1:                        {self.redirected_s1} ({self.redirected_s1 * per_hour:.4f} AMRs/hour)")
        print(f"  - At S2:                        {self.redirected_s2} ({self.redirected_s2 * per_hour:.4f} AMRs/hour)")
        print(f"  - At S3:                        {self.redirected_s3} ({self.redirected_s3 * per_hour:.4f} AMRs/hour)")
        print(f"  - Total:                        {total_redirected} ({total_redirected * per_hour:.4f} AMRs/hour)")

        # print for every station: utilization, average queue length, average wait in buffer, and completed AMRs
        station_list = [("S1", self.area_s1, self.area_q1, self.total_delay_s1, self.s1_started, self.s1_completed),
                        ("S2", self.area_s2, self.area_q2, self.total_delay_s2, self.s2_started, self.s2_completed),
                        ("S3", self.area_s3, self.area_q3, self.total_delay_s3, self.s3_started, self.s3_completed)]

        for name, area_s, area_q, delay, started, completed in (station_list):
            print("-" * 70)
            print(f"WORKSTATION {name} PERFORMANCE:")
            print(f"  - Utilization:                  {area_s / self.sim_time * 100:.2f}%")
            print(f"  - Avg Queue Length:             {area_q / self.sim_time:.4f} AMRs")
            print(f"  - Avg Wait in Buffer:           {delay / started if started > 0 else 0.0:.4f} minutes")
            print(f"  - Completed AMRs:               {completed}")

        # print for all stations in the hub: average total wait and average time in hub
        print("-" * 70)
        print("OVERALL, FOR AMRs THAT COMPLETED SERVICE IN THE HUB:")
        print(f"  - Avg Total Wait:               {self.total_delay_hub / self.hub_completed:.4f} minutes")
        print(f"  - Avg Time in Hub:              {self.total_time_in_hub / self.hub_completed:.4f} minutes")
        print("=" * 70)

if __name__ == "__main__":
    sim = AmrSimulation(
        arrival_rate=45.0,
        buffer_capacity=3,
        sim_duration=100000.0,
        seed=42
    )
    sim.run()
