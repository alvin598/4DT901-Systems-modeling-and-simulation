from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import stats

# all the headers in the .csv file
CLOUD_COMPUTE_REQUESTS_HEADERS = ["request_id", "arrival_time_min", "processed_time_min", "compute_cluster","network_latency_min"]

csv_path = Path(__file__).parent / "data" / "cloud_compute_requests.csv"
input_data = pd.read_csv(csv_path)

request_ids = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[0]].to_numpy()
arrival_times_min = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[1]].to_numpy()
processed_times_min = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[2]].to_numpy()
compute_clusters = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[3]].to_numpy()
network_latencies_min = input_data[CLOUD_COMPUTE_REQUESTS_HEADERS[4]].to_numpy()

inter_arrival = np.diff(arrival_times_min)
services = processed_times_min - arrival_times_min

bin_count = math.ceil(math.sqrt(len(services)))

# Erlang shape parameter k = (mean^2) / variance
k = max(1, round(services.mean() ** 2 / services.var(ddof=1)))

# gamma fitted with maximum likelihood, location fixed at 0
gamma_shape, _, gamma_scale = stats.gamma.fit(services, floc=0)

# KS test for goodness of fit
p_arr = stats.kstest(inter_arrival, "expon", args=(0, inter_arrival.mean())).pvalue
p_srv = stats.kstest(services, "expon", args=(0, services.mean())).pvalue
p_srv_erl = stats.kstest(services, "erlang", args=(k, 0, services.mean() / k)).pvalue
p_srv_gamma = stats.kstest(services, "gamma", args=(gamma_shape, 0, gamma_scale)).pvalue

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# interarrival histogram with the fitted exponential pdf
axes[0].hist(inter_arrival, bins=bin_count, density=True)
x = np.linspace(0, inter_arrival.max(), 400)
axes[0].plot(x, stats.expon.pdf(x, scale=inter_arrival.mean()), "r-", label="exponential")
axes[0].set_title(f"Interarrival\nExponential: {p_arr:.2f}")
axes[0].set_xlabel("minutes")

# service time histogram with the fitted exponential, erlang and gamma pdf
axes[1].hist(services, bins=bin_count, density=True)
x = np.linspace(0, services.max(), 400)
axes[1].plot(x, stats.expon.pdf(x, scale=services.mean()), "r-", label="exponential")
axes[1].plot(x, stats.erlang.pdf(x, a=k, scale=services.mean() / k), "g-", label=f"erlang (k={k})")
axes[1].plot(x, stats.gamma.pdf(x, a=gamma_shape, scale=gamma_scale), "y-", label=f"gamma (shape={gamma_shape:.2f})")
axes[1].set_title(f"Service time\nExponential: {p_srv:.6f}, Erlang: {p_srv_erl:.6f}, Gamma: {p_srv_gamma:.6f}")
axes[1].set_xlabel("minutes")

statistics = {
    "mean_interarrival": inter_arrival.mean(),
    "std_interarrival": inter_arrival.std(ddof=1),
    "var_interarrival": inter_arrival.var(ddof=1),
    "n_service": len(services),
    "mean_service_time": services.mean(),
    "std_service_time": services.std(ddof=1),
    "var_service_time": services.var(ddof=1),
    "gamma_shape": gamma_shape,
}

# create a pandas series and print the statistics for the combined data
statistics_table = pd.Series(statistics)
print(statistics_table.round(4))

plt.tight_layout()
plt.show()