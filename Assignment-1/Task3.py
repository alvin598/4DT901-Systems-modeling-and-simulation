from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import stats

# all the headers in the .csv file
CLOUD_COMPUTE_REQUESTS_HEADERS = ["request_id", "arrival_time_min", "processed_time_min", "compute_cluster","network_latency_min"]

# csv_path = Path(__file__).with_name("data/cloud_compute_requests.csv")
input_data = pd.read_csv("Assignment-1/data/cloud_compute_requests.csv")

request_ids = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[0]].to_numpy()
arrival_times_min = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[1]].to_numpy()
processed_times_min = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[2]].to_numpy()
compute_clusters = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[3]].to_numpy()
network_latencies_min = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[4]].to_numpy()


C1_Media = []
C2_Ai = []
C3_Database = []

previouse_arrival_time = {}

for arrival_time, processed_time, cluster in zip(arrival_times_min, processed_times_min, compute_clusters):

    # for the first iteration we set the interarrival time to the arrival time
    if cluster in previouse_arrival_time:
        interarrival_time = arrival_time - previouse_arrival_time[cluster]
    else:
        interarrival_time = arrival_time

    service_time = processed_time - arrival_time
    match cluster:
        case "C1_Media":
            C1_Media.append([interarrival_time, service_time])
        case "C2_AI":
            C2_Ai.append([interarrival_time, service_time])
        case "C3_Database":
            C3_Database.append([interarrival_time, service_time])
        case _:
            raise ValueError(f"Unknown compute cluster: {cluster}")
    
    previouse_arrival_time[cluster] = arrival_time



C1_Media = np.array(C1_Media)
C2_Ai = np.array(C2_Ai)
C3_Database = np.array(C3_Database)


clusters = [("C1_Media", C1_Media), ("C2_AI", C2_Ai), ("C3_Database", C3_Database)]

fig, axes = plt.subplots(3, 2, figsize=(10, 9))

# plot all three cluster's pdf for both arrival time and service time
for i, (name, cluster_data) in enumerate(clusters):
    arrivals = cluster_data[:, 0]
    services = cluster_data[:, 1]
    bin_count = math.ceil(math.sqrt(len(cluster_data)))

    # Erlang shape parameter k = (mean^2) / variance, only for service since all interarrival passed ks test for exponential distribution
    k = max(1, round(services.mean() ** 2 / services.var()))

    # plot histograms
    axes[i, 0].hist(arrivals, bins=bin_count, density=True)
    axes[i, 0].set_title(f"{name} - interarrival")

    # plot the probability distrubutions for erlang and exponential pdf
    x = np.linspace(0, arrivals.max(), 400)
    axes[i, 0].plot(x, stats.expon.pdf(x, scale=arrivals.mean()), "r-")

    # do the same for service time
    axes[i, 1].hist(services, bins=bin_count, density=True)
    axes[i, 1].set_title(f"{name} - service time")

    # plot the probability distrubutions for erlang and exponential pdf
    x = np.linspace(0, arrivals.max(), 400)
    axes[i, 1].plot(x, stats.expon.pdf(x, scale=arrivals.mean()), "r-")
    axes[i, 1].plot(x, stats.erlang.pdf(x, a=k, scale=arrivals.mean() / k), "g-")


    inter_arrival = (cluster_data[:, 0])
    services = cluster_data[:, 1]

    # KS test for goodness of fit
    p_arr = stats.kstest(inter_arrival, "expon", args=(0, inter_arrival.mean())).pvalue
    p_srv = stats.kstest(services, "expon", args=(0, services.mean())).pvalue
    p_srv_erl = stats.kstest(services, "erlang", args=(k, 0, services.mean() / k)).pvalue
    print(
        f"{name}: Exponential: "
        f"Interarrival: {p_arr}, Service: {p_srv}"
    )
    print(f"{name}: Erlang (k={k}): Service: {p_srv_erl}")

plt.tight_layout()
plt.show()