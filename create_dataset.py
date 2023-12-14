from utils.data_reader import SinD
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd
import os

ROOT = os.getcwd()


def calculate_angle_between_vectors(v1, v2):
    # Convert the input lists to NumPy arrays if they aren't already
    v1, v2 = np.array(v1), np.array(v2)

    # Calculate the dot product
    dot_product = np.dot(v1, v2)

    # Calculate the norms of each vector
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip for floating point precision
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_direction_change(x, y):
    angles = []
    for i in range(2, len(x)):
        vec1 = np.array([x[i-1] - x[i-2], y[i-1] - y[i-2]])
        vec2 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        angle = calculate_angle_between_vectors(vec1, vec2)
        angles.append(angle)
        
    return np.nan_to_num(angles)  # Replace NaNs with 0


def is_crossing_street(pedestrian_path, map):
    return pedestrian_path.intersects(map.road_poly) or pedestrian_path.intersects(map.crosswalk_poly) \
        or pedestrian_path.intersects(map.intersection_poly) or pedestrian_path.intersects(map.gap_poly)


def is_illegal_crossing(pedestrian_path, map):
    return (pedestrian_path.intersects(map.road_poly) or pedestrian_path.intersects(map.intersection_poly))


def has_made_turn(x, y, angle_threshold=30):
    # Calculate if direction has changed between the first and last point more than threshold (degrees)
    # return any(angle > angle_threshold for angle in angles)
    start_vector = np.array([x[1] - x[0], y[1] - y[0]])
    end_vector = np.array([x[-1] - x[-2], y[-1] - y[-2]])

    angle = calculate_angle_between_vectors(start_vector, end_vector)
    return angle > angle_threshold


def check_point_location(point):
    point_location = ''
    for i in range(0, 4):
        if point.within(map.sidewalk_poly[i]) or point.touches(map.sidewalk_poly[i]):
            point_location = f'sidewalk{i}'

    if not point.within(map.sidewalk_poly) and not point.touches(map.sidewalk_poly):
        if point.within(map.gap_poly) or point.touches(map.gap_poly):
            point_location = f'gap'
        if point.within(map.road_poly) or point.touches(map.road_poly):
            point_location = f'road'
        if point.within(map.intersection_poly) or point.touches(map.intersection_poly):
            point_location = f'inter'
        if point.within(map.crosswalk_poly) or point.touches(map.crosswalk_poly):
            point_location = f'crosswalk'

    return point_location


def count_different_locations(x, y):
    data = {'sidewalk0__count': 0, 'sidewalk1__count': 0, 'sidewalk2__count': 0, 'sidewalk3__count': 0, 'gap__count': 0, 'road__count': 0,
               'inter__count': 0, 'crosswalk__count': 0, '': 0}
    for x_point, y_point in zip(x, y):
        data[check_point_location(Point(x_point, y_point)) + '__count'] +=1

    del data['']
    return data


def get_recorded_features(pedestrian_id, dataset, input_len=90):
    # input length: moving window when data in arrays are povided (not related to chunk size)

    if type(dataset) is not dict:
        data = []
        idx = 0
        for _ in range(6):
            data.append(pd.DataFrame(dataset).iloc[pedestrian_id,:][idx:idx+input_len])
            idx += input_len
            
        x, y, vx, vy, ax, ay = data[0].to_numpy(), data[1].to_numpy(), \
            data[2].to_numpy(), data[3].to_numpy(), data[4].to_numpy(), data[5].to_numpy()
    else:
        x, y, vx, vy, ax, ay = dataset[pedestrian_id]['x'].to_numpy(), dataset[pedestrian_id]['y'].to_numpy(), \
            dataset[pedestrian_id]['vx'].to_numpy(), dataset[pedestrian_id]['vy'].to_numpy(), dataset[pedestrian_id]['ax'].to_numpy(), dataset[pedestrian_id]['ay'].to_numpy()

    return  x, y, vx, vy, ax, ay


def get_data_where_pedestrians_move(attributes, threshold: float = 0.5, velocity_filter: bool = True):
    v = np.linalg.norm(list(zip(attributes['vx'], attributes['vy'])), axis=1)
    _id = np.where(v >= threshold) if velocity_filter else np.where(v >= -1) # make sure pedestrians are moving
    new_attributes = {}

    for key in attributes.keys():
        new_attributes[key] = attributes[key].iloc[_id]

    return new_attributes


def split_pedestrian_data(data, remove_staionary_data: bool = False, chunk_size: float =90):
    """
    Split pedestrian data into chunks.

    :param data: Dictionary of pedestrian data.
    :param chunk_size: Maximum number of steps in each chunk.
    :return: Dictionary with split data.
    """
    MIN_DATA_INCLUDED = 5
    split_data = {}
    idx = 0
    pedestrian_ids = []


    for pedestrian_id, attributes in data.items():
        # Determine the number of chunks needed for this pedestrian
        new_attributes = attributes
        if remove_staionary_data:
            new_attributes = get_data_where_pedestrians_move(attributes)

        extra_data = len(new_attributes['x']) % chunk_size
        flag_extra_data = extra_data > MIN_DATA_INCLUDED
        num_chunks = len(new_attributes['x']) // chunk_size + (1 if flag_extra_data else 0)

        for i in range(num_chunks):
            split_data[idx] = {}
            max_size = chunk_size + (extra_data if ((not flag_extra_data) and (i == num_chunks-1)) else 0)
            for attr, values in new_attributes.items():
                start_index = i * chunk_size
                end_index = start_index + max_size
                split_data[idx][attr] = values[start_index:end_index]
            
            pedestrian_ids.append(pedestrian_id)
            idx += 1

    return split_data, pd.DataFrame(pedestrian_ids)


def create_dataframe(dataset, input_len=90):
    df = pd.DataFrame()
    for key in range(len(dataset)):
        x, y, vx, vy, ax, ay = get_recorded_features(key, dataset, input_len)

        speed = pd.DataFrame(np.sqrt(np.square(vx) + np.square(vy)))
        acceleration = pd.DataFrame(np.sqrt(np.square(ax) + np.square(ay)))
        direction_change = pd.DataFrame(calculate_direction_change(x, y))

        proximity_to_crosswalk__start = Point(x[0], y[0]).distance(map.crosswalk_poly)
        proximity_to_crosswalk__last = Point(x[-1], y[-1]).distance(map.crosswalk_poly)

        proximity_to_road__start = Point(x[0], y[0]).distance(map.road_poly)
        proximity_to_road__last = Point(x[-1], y[-1]).distance(map.road_poly)
        proximity_to_road__min = min([Point(x[i], y[i]).distance(map.road_poly) for i in range(len(x))])

        proximity_to_inter__start = Point(x[0], y[0]).distance(map.intersection_poly)
        proximity_to_inter__last = Point(x[-1], y[-1]).distance(map.intersection_poly)
        proximity_to_inter__min = min([Point(x[i], y[i]).distance(map.intersection_poly) for i in range(len(x))])

        pedestrian_path = LineString(zip(x, y))
        crossing_feature = is_crossing_street(pedestrian_path, map)
        illegal_crossing_feature = is_illegal_crossing(pedestrian_path, map)
        turn_feature = has_made_turn(x, y)

        # acceleration
        new_row_data = {}
        new_row_data['acc__min'] = acceleration.min()[0]
        new_row_data['acc__max'] = acceleration.max()[0]
        new_row_data['acc__avg'] = acceleration.mean()[0]
        new_row_data['acc__median'] = acceleration.quantile(.5)[0]
        new_row_data['acc__first_quantile'] = acceleration.quantile(.25)[0]
        new_row_data['acc__last_quantile'] = acceleration.quantile(.75)[0]
        new_row_data['acc__start'] = acceleration.iloc[0][0]
        new_row_data['acc__last'] = acceleration.iloc[-1][0]

        # speed
        new_row_data['speed__min'] = speed.min()[0]
        new_row_data['speed__max'] = speed.max()[0]
        new_row_data['speed__avg'] = speed.mean()[0]
        new_row_data['speed__median'] = speed.quantile(.5)[0]
        new_row_data['speed__first_quantile'] = speed.quantile(.25)[0]
        new_row_data['speed__last_quantile'] = speed.quantile(.75)[0]
        new_row_data['speed__start'] = speed.iloc[0][0]
        new_row_data['speed__last'] = speed.iloc[-1][0]

        # direction change
        new_row_data['direction_change__min'] = direction_change.min()[0]
        new_row_data['direction_change__max'] = direction_change.max()[0]
        new_row_data['direction_change__avg'] = direction_change.mean()[0]
        new_row_data['direction_change__median'] = direction_change.quantile(.5)[0]
        new_row_data['direction_change__first_quantile'] = direction_change.quantile(.25)[0]
        new_row_data['direction_change__last_quantile'] = speed.quantile(.75)[0]
        new_row_data['direction_change__start'] = direction_change.iloc[0][0]
        new_row_data['direction_change__last'] = direction_change.iloc[-1][0]

        # proximity_to_crosswalk
        new_row_data['proximity_to_crosswalk__start'] = proximity_to_crosswalk__start
        new_row_data['proximity_to_crosswalk__last'] = proximity_to_crosswalk__last

        # proximity_to_road
        new_row_data['proximity_to_road__start'] = proximity_to_road__start
        new_row_data['proximity_to_road__last'] = proximity_to_road__last
        new_row_data['proximity_to_road__min'] = proximity_to_road__min

        # proximity_to_intersection
        new_row_data['proximity_to_inter__start'] = proximity_to_inter__start
        new_row_data['proximity_to_inter__last'] = proximity_to_inter__last
        new_row_data['proximity_to_inter__min'] = proximity_to_inter__min

        # Action Features
        new_row_data['crossing'] = crossing_feature
        new_row_data['illegal_crossing'] = illegal_crossing_feature
        new_row_data['turning'] = turn_feature

        # Starting Point Feature
        new_row_data['starting_point'] = check_point_location(Point(x[0], y[0])) + '__start'
        new_row_data['ending_point'] = check_point_location(Point(x[-1], y[-1])) + '__end'
        new_row_data.update( count_different_locations(x, y))

        new_row_df = pd.DataFrame([new_row_data])
        df = pd.concat([df, new_row_df], ignore_index=True)
    
    return df

def create_dataframes_with_different_chunk_sizes(start=5, end=90, step=5):

    for chunk_size in range(start, end, step):
        (dataset, pedestrian_ids), data_type = split_pedestrian_data(sind.pedestrian_data, remove_staionary_data=False, chunk_size=chunk_size), f'__full_splitted_data_{chunk_size}'
    
        # df = create_dataframe(dataset)
        # df.to_csv(ROOT + f"/dataset_created{data_type}.csv", encoding='utf-8', index=False)
        pedestrian_ids.to_csv(ROOT + f"/pedestrian_ids_{data_type}.csv", encoding='utf-8', index=False)


def find_pedestrians_with_high_velocity():
    ids_array = []
    pedestrian_data = sind.pedestrian_data
    for key, data in pedestrian_data.items():
        v = np.sqrt(np.square(data['vx'] ) + np.square(data['vy'] ))
        if any(v > 5.5): ids_array.append(key)

    ids_array
    


if __name__ == "__main__":
    sind = SinD()
    map = sind.map
    # input_len = 90
    # # TODO Keep 90 e.g steps (experiment with that number) without overlaping and keep the last value as a feature of all the variables (e.g direction, speed, acc etc.)
    # # ADD also heading
    # # dataset, data_type = sind.data(input_len=input_len), ''
    # # dataset, data_type = sind.pedestrian_data, '__full'
    # # dataset, data_type = split_pedestrian_data(sind.pedestrian_data), '__full_splitted'
    # dataset, data_type = split_pedestrian_data(sind.pedestrian_data, remove_staionary_data=True), '__full_splitted_without_stationary_data'
    
    # df = create_dataframe(dataset, input_len)
    # df.to_csv(ROOT + f"/dataset_created{data_type}.csv", encoding='utf-8', index=False)
    create_dataframes_with_different_chunk_sizes(start=90, end=95)
