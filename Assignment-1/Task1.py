from pathlib import Path

import pandas as pd

from scipy import stats

csv_path = Path(__file__).parent / "data" / "own_experiment_data.csv"
input_data = pd.read_csv(csv_path)

interarrival_times = input_data["interarrival_time_s"].to_numpy(dtype=float)
service_times = input_data["service_time_s"].to_numpy(dtype=float)

samples = [("Interarrival time", interarrival_times), ("Service time", service_times)]


# print the parameters and the ks test for interarrival times
for name, data in samples:
    mean = data.mean()
    p_value = stats.kstest(data, "expon", args=(0, mean)).pvalue

    print(f"{name}")
    print(f"  mean     = {mean:.2f} s")
    print(f"  std      = {data.std(ddof=1):.2f} s")
    print(f"  variance = {data.var(ddof=1):.2f} s^2")
    print(f"  KS test p values: {p_value}\n")
