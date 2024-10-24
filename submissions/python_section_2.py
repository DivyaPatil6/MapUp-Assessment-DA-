import pandas as pd
import numpy as np
from datetime import time, timedelta, datetime, date

def calculate_distance_matrix(df)->pd.DataFrame():
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    np.fill_diagonal(distance_matrix.values, 0)
    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance 
    # Applyig Floyd-Warshall algorithm
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

csv = '../datasets/dataset-2.csv' 
df = pd.read_csv(csv)
distance_matrix = calculate_distance_matrix(df)
print("9. Distance Matrix:")
print(distance_matrix)


def unroll_distance_matrix(df)->pd.DataFrame():
    data = []
    for i in df.columns:
        for j in df.columns:
            if i != j: 
                data.append({'id_start': i, 'id_end': j, 'distance': df.loc[i, j]})
    unrolled_df = pd.DataFrame(data)

    return unrolled_df
unrolled_df = unroll_distance_matrix(distance_matrix)
print("\n10. Unrolled Distance:")
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    reference_df = df[df['id_start'] == reference_id]
    reference_avg_distance = reference_df['distance'].mean()

    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1

    average_distances = df.groupby('id_start')['distance'].mean().reset_index()

    #Filtering IDs whose average distance lies within the 10% range
    within_threshold_df = average_distances[
        (average_distances['distance'] >= lower_bound) & 
        (average_distances['distance'] <= upper_bound)
    ]

    sorted_ids = within_threshold_df['id_start'].sort_values().tolist()

    return sorted_ids

unrolled_df = unroll_distance_matrix(distance_matrix)

# Sample output for ref_id=1001400
reference_id = 1001400
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print("\n11. IDs within 10% of reference ID's average distance:")
print(result_ids)


def calculate_toll_rate(df)->pd.DataFrame():
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    return df

unrolled_df = unroll_distance_matrix(distance_matrix)
toll_rate_df = calculate_toll_rate(unrolled_df)
print("\n12. Toll Rates:")
print(toll_rate_df[['id_start','id_end', 'moto', 'car', 'rv', 'bus', 'truck']])





days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
start_times = [
    time(9, 0),   # Monday
    time(10, 0),  # Tuesday
    time(8, 0),   # Wednesday
    time(11, 0),  # Thursday
    time(12, 0),  # Friday
    time(14, 0),  # Saturday
    time(15, 0)   # Sunday
]
end_times = [
    time(10, 0),  # Monday
    time(11, 0),  # Tuesday
    time(9, 0),   # Wednesday
    time(12, 0),  # Thursday
    time(13, 0),  # Friday
    time(16, 0),  # Saturday
    time(17, 0)   # Sunday
]


toll_rate_df['start_day'] = days * (len(toll_rate_df) // len(days)) + days[:len(toll_rate_df) % len(days)]
toll_rate_df['start_time'] = start_times * (len(toll_rate_df) // len(start_times)) + start_times[:len(toll_rate_df) % len(start_times)]
toll_rate_df['end_day'] = toll_rate_df['start_day']
toll_rate_df['end_time'] = end_times * (len(toll_rate_df) // len(end_times)) + end_times[:len(toll_rate_df) % len(end_times)]

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    time_intervals = [
        (time(0, 0), time(10, 0), 0.8), 
        (time(10, 0), time(18, 0), 1.2), 
        (time(18, 0), time(23, 59, 59), 0.8)
    ]
    weekend_discount = 0.7
    
    for start_day in weekdays:
        for start_time, end_time, discount in time_intervals:
            mask = (df['start_day'] == start_day) & (df['start_time'] >= start_time) & (df['end_time'] <= end_time)
            df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount

    for start_day in weekends:
        mask = df['start_day'] == start_day
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= weekend_discount

    return df
time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)
time_based_toll_rate_df = time_based_toll_rate_df[['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']]
print("\n13. Time-Based Toll Rates:")
print(time_based_toll_rate_df)