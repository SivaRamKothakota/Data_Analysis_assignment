import pandas as pd
import numpy as np

def generate_car_matrix(df):

    df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    np.fill_diagonal(df.values, 0)

    return df


def get_type_count(df):

    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = np.select(conditions, choices, default='Unknown')


    type_counts = df['car_type'].value_counts().to_dict()


    dict = {key: type_counts[key] for key in sorted(type_counts.keys())}

    return dict


def get_bus_indexes(df):

    bus_mean = df['bus'].mean()

    l = df[df['bus'] > 2 * bus_mean].index.tolist()

    l.sort()

    return l

def filter_routes(df):

    avg_truck_by_route = df.groupby('route')['truck'].mean()


    list = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()


    list.sort()

    return list

def multiply_matrix(matrix):

    matrix = matrix.apply(lambda x:x.map(lambda y: y * 0.75 if y > 20 else y * 1.25))


    matrix = matrix.round(1)

    return matrix


def time_check(df):

    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'],format='%Y-%m-%d %H:%M',errors='coerce')


    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'],format='%Y-%m-%d %H:%M',errors='coerce')

    df['duration'] = df['end_datetime'] - df['start_datetime']

    full_24_hour = df['duration'].dt.total_seconds() >= 86400
    all_7_days = df['start_datetime'].dt.dayofweek.nunique() == 7

    df = df.groupby(['id', 'id_2']).apply(lambda x: all(full_24_hour.loc[x.index]) and all_7_days.loc[x.index[0]])

    return df

df = pd.read_csv(r'C:\Users\SIVARAM PC\Desktop\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv')
sample_result = generate_car_matrix(df)
print("Question 1:\n",sample_result)

result = get_type_count(df)
print("Question 2:\n",result)

result = get_bus_indexes(df)
print(" Question 3:\n",result)

result = filter_routes(df)
print("Question 4:\n",result)

modified_result = multiply_matrix(sample_result)  # Assuming sample_result is the matrix generated earlier
print("Question 5:\n",modified_result)

df1 = pd.read_csv(r'C:\Users\SIVARAM PC\Desktop\MapUp-Data-Assessment-F-main\datasets\dataset-2.csv')
boolean_series = time_check(df1)
print("Question 6:\n",boolean_series)