from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import stats


csv_path = Path(__file__).with_name("cloud_compute_requests.csv")
input_data = pd.read_csv(csv_path)

request_ids = input_data["request_id"].to_numpy()
arrival_times_min = input_data["arrival_time_min"].to_numpy()
processed_times_min = input_data["processed_time_min"].to_numpy()
compute_clusters = input_data["compute_cluster"].to_numpy()
network_latencies_min = input_data["network_latency_min"].to_numpy()


C1_Media = []
C2_Ai = []
C3_Database = []

previouse_arrival_time = {}

for arrival_time, processed_time, cluster in zip(arrival_times_min, processed_times_min, compute_clusters):

    # for the first iteration we set the interarrival time to the arrival time
    if cluster in previouse_arrival_time:                                    # changed
        interarrival_time = arrival_time - previouse_arrival_time[cluster]   # changed    interarrival_time = arrival_time - previouse_arrival_time
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

# d, p = stats.kstest(C1_Media, "expon")

# print(p)


clusters = [("C1_Media", C1_Media), ("C2_AI", C2_Ai), ("C3_Database", C3_Database)]

fig, axes = plt.subplots(3, 2, figsize=(10, 9))

for i, (name, cluster_data) in enumerate(clusters):
    arrivals = cluster_data[:, 0]   # column 0 = interarrival time
    services = cluster_data[:, 1]   # column 1 = service time
    bin_count = math.ceil(math.sqrt(len(cluster_data)))

    # Erlang shape parameter k = (mean^2) / variance, only for service since all interarrival passed ks test for exponential distribution
    k = max(1, round(services.mean() ** 2 / services.var()))

    # plot histograms and fitted distributions
    axes[i, 0].hist(arrivals, bins=bin_count, density=True)
    axes[i, 0].set_title(f"{name} – interarrival")
    x = np.linspace(0, arrivals.max(), 400)
    axes[i, 0].plot(x, stats.expon.pdf(x, scale=arrivals.mean()), "r-")

    
    axes[i, 1].hist(services, bins=bin_count, density=True)
    axes[i, 1].set_title(f"{name} – service time")
    x = np.linspace(0, services.max(), 400)
    axes[i, 1].plot(x, stats.expon.pdf(x, scale=services.mean()), "r-")
    axes[i, 1].plot(x, stats.erlang.pdf(x, a=k, scale=services.mean() / k), "g--")

    inter_arrival = (cluster_data[:, 0])   # gaps between consecutive arrivals
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