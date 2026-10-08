from pathlib import Path

import pandas as pd
import numpy as np

csv_path = Path(__file__).with_name("cloud_compute_requests.csv")
input_data = pd.read_csv(csv_path)

request_ids = input_data["request_id"].to_numpy()
arrival_times_min = input_data["arrival_time_min"].to_numpy()
processed_times_min = input_data["processed_time_min"].to_numpy()
compute_clusters = input_data["compute_cluster"].to_numpy()
network_latencies_min = input_data["network_latency_min"].to_numpy()


C1_Media = np.array()
C2_Ai = []
C3_Database = []

for arrival_time, processed_time, cluster in zip(arrival_times_min, processed_times_min, compute_clusters):
    # print(arrival_time + processed_time)


    service_time = processed_time - arrival_time
    match cluster:
        case "C1_Media":
            C1_Media.append(service_time)
        case "C2_AI":
            C2_Ai.append(service_time)
        case "C3_Database":
            C3_Database.append(service_time)
        case _:
            raise ValueError(f"Unknown compute cluster: {cluster}")

    print(service_time)