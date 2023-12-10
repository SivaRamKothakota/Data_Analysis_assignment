import pandas as pd
import numpy as np
from datetime import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def calculate_distance_matrix(df):

    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(np.zeros((len(unique_ids), len(unique_ids))), index=unique_ids, columns=unique_ids)


    for index, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        distance = row['distance']


        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance

    for i in unique_ids:
        for j in unique_ids:
            for k in unique_ids:
                if distance_matrix.at[i, j] == 0 and i != j and i != k and j != k:
                    if distance_matrix.at[i, k] != 0 and distance_matrix.at[k, j] != 0:
                        distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix


def unroll_distance_matrix(distance_matrix):

    unique_ids = distance_matrix.index

    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    for i in range(len(unique_ids)):
        for j in range(i + 1, len(unique_ids)):
            id_start = unique_ids[i]
            id_end = unique_ids[j]
            distance = distance_matrix.at[id_start, id_end]

            unrolled_df = unrolled_df._append({'id_start': id_start, 'id_end': id_end, 'distance': distance},
                                             ignore_index=True)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id):

    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()


    threshold_min = reference_avg_distance - (reference_avg_distance * 0.1)
    threshold_max = reference_avg_distance + (reference_avg_distance * 0.1)


    filtered_ids = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = filtered_ids[
        (filtered_ids['distance'] >= threshold_min) & (filtered_ids['distance'] <= threshold_max)]

    result_df = filtered_ids.sort_values(by='id_start')
    return result_df


def calculate_toll_rate(df):

    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }


    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient


    df = df.drop(columns='distance')

    return df


def calculate_time_based_toll_rates(df):

    time_ranges = {
        'weekday_early': (time(0, 0), time(10, 0)),
        'weekday_midday': (time(10, 0), time(18, 0)),
        'weekday_evening': (time(18, 0), time(23, 59, 59)),
        'weekend_all_day': (time(0, 0), time(23, 59, 59))
    }

    discount_factors = {
        'weekday_early': 0.8,
        'weekday_midday': 1.2,
        'weekday_evening': 0.8,
        'weekend_all_day': 0.7
    }


    result_df = df.copy()


    result_df['start_day'] = 'Monday'
    result_df['start_time'] = time(0, 0)
    result_df['end_day'] = 'Sunday'
    result_df['end_time'] = time(23, 59, 59)


    for time_range, (start_time, end_time) in time_ranges.items():
        mask = (result_df['start_time'] <= end_time) & (result_df['end_time'] >= start_time)

        if 'weekday' in time_range:
            result_df.loc[mask, 'discount_factor'] = discount_factors[time_range]
        else:
            result_df.loc[mask, 'discount_factor'] = discount_factors['weekend_all_day']


    for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
        result_df[vehicle_type] = result_df[vehicle_type] * result_df['discount_factor']


    result_df = result_df.drop(columns=['discount_factor'])

    return result_df


data = pd.read_csv(r'C:\Users\SIVARAM PC\Desktop\MapUp-Data-Assessment-F-main\datasets\dataset-3.csv')


resulting_distance_matrix = calculate_distance_matrix(data)
print("Distance Matrix Calculation:\n",resulting_distance_matrix)


unrolled_data = unroll_distance_matrix(resulting_distance_matrix)
print("Unroll Distance Matrix:\n",unrolled_data)


reference_value = 1001404
resulting_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_data, reference_value)
print("Finding IDs within Percentage Threshold:\n",resulting_ids_within_threshold)


result_with_toll_rates = calculate_toll_rate(unrolled_data)
print("Calculate Toll Rate:\n",result_with_toll_rates)

time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rates)
print("Calculate Time-Based Toll Rate:\n",time_based_toll_rates)