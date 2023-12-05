from utils.data_reader import SinD
from shapely.geometry import Point, LineString

import numpy as np
import pandas as pd
import os

ROOT = os.getcwd()


def calculate_angle_between_vectors(v1, v2):
    # TODO test thoroughly, recreate the 45 degrees angle
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def calculate_turning_angles(path):
    """Calculate turning angles along a path."""
    angles = []
    for i in range(1, len(path.coords) - 1):
        p0 = np.array(path.coords[i - 1])
        p1 = np.array(path.coords[i])
        p2 = np.array(path.coords[i + 1])

        vec1 = p1 - p0
        vec2 = p2 - p1

        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            continue  # Skip if either vector is zero to avoid division by zero

        angle = calculate_angle_between_vectors(vec1, vec2)
        angles.append(angle)

    return angles

def calculate_direction_change(x, y):
    angles = []
    for i in range(2, len(x)):
        vec1 = [x[i-1] - x[i-2], y[i-1] - y[i-2]]
        vec2 = [x[i] - x[i-1], y[i] - y[i-1]]
        angle = calculate_angle_between_vectors(vec1, vec2)
        angles.append(angle)
    return np.nan_to_num(angles)  # Replace NaNs with 0

def is_crossing_street(pedestrian_path, map):
    return pedestrian_path.intersects(map.road_poly) or pedestrian_path.intersects(map.crosswalk_poly) \
        or pedestrian_path.intersects(map.intersection_poly) or pedestrian_path.intersects(map.gap_poly)

def is_illegal_crossing(pedestrian_path, map):
    return (pedestrian_path.intersects(map.road_poly) or pedestrian_path.intersects(map.intersection_poly))

def has_made_turn(pedestrian_path, angle_threshold=30):
    # Assuming pedestrian_path is a LineString
    # Calculate the turning angles along the path
    angles = calculate_turning_angles(pedestrian_path)
    return any(angle > angle_threshold for angle in angles)

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


def get_recorded_features(pedestrian_id, dataset, input_len):
    if type(dataset) is not dict:
        data = []
        idx = 0
        for _ in range(6):
            data.append(pd.DataFrame(dataset).iloc[pedestrian_id,:][idx:idx+input_len])
            idx += input_len
            
        x, y, vx, vy, ax, ay = data[0].reset_index(drop=True), data[1].reset_index(drop=True), \
            data[2].reset_index(drop=True), data[3].reset_index(drop=True), data[4].reset_index(drop=True), data[5].reset_index(drop=True)

    else:    
        x, y, vx, vy, ax, ay = dataset[pedestrian_id]['x'].reset_index(drop=True), dataset[pedestrian_id]['y'].reset_index(drop=True), \
            dataset[pedestrian_id]['vx'].reset_index(drop=True), dataset[pedestrian_id]['vy'].reset_index(drop=True), dataset[pedestrian_id]['ax'].reset_index(drop=True), dataset[pedestrian_id]['ay'].reset_index(drop=True)

    return  x, y, vx, vy, ax, ay

def create_dataframe(dataset, input_len):
    df = pd.DataFrame()
    for pedestrian_id in range(len(dataset)):
        x, y, vx, vy, ax, ay = get_recorded_features(pedestrian_id, dataset, input_len)

        speed = pd.DataFrame(np.sqrt(np.square(vx.to_numpy()) + np.square(vy.to_numpy())))
        acceleration = pd.DataFrame(np.sqrt(np.square(ax.to_numpy()) + np.square(ay.to_numpy())))
        direction_change = pd.DataFrame(calculate_direction_change(x, y))
        proximity_to_crosswalk = Point(x[0], y[0]).distance(map.crosswalk_poly)
        proximity_to_road_start = Point(x[0], y[0]).distance(map.road_poly)
        proximity_to_road_min = min([Point(x[i], y[i]).distance(map.road_poly) for i in range(len(x))])
        proximity_to_inter_start = Point(x[0], y[0]).distance(map.intersection_poly)
        proximity_to_inter_min = min([Point(x[i], y[i]).distance(map.intersection_poly) for i in range(len(x))])

        pedestrian_path = LineString(zip(np.array(x), np.array(y))) 
        crossing_feature = is_crossing_street(pedestrian_path, map)
        illegal_crossing_feature = is_illegal_crossing(pedestrian_path, map)
        turn_feature = has_made_turn(pedestrian_path)

        # acceleration
        new_row_data = {}
        new_row_data['acc__min'] = acceleration.min()[0]
        new_row_data['acc__max'] = acceleration.max()[0]
        new_row_data['acc__avg'] = acceleration.mean()[0]
        new_row_data['acc__median'] = acceleration.quantile(.5)[0]
        new_row_data['acc__first_quantile'] = acceleration.quantile(.25)[0]
        new_row_data['acc__last_quantile'] = acceleration.quantile(.75)[0]

        # speed
        new_row_data['speed__min'] = speed.min()[0]
        new_row_data['speed__max'] = speed.max()[0]
        new_row_data['speed__avg'] = speed.mean()[0]
        new_row_data['speed__median'] = speed.quantile(.5)[0]
        new_row_data['speed__first_quantile'] = speed.quantile(.25)[0]
        new_row_data['speed__last_quantile'] = speed.quantile(.75)[0]

        # direction change
        new_row_data['direction_change__min'] = direction_change.min()[0]
        new_row_data['direction_change__max'] = direction_change.max()[0]
        new_row_data['direction_change__avg'] = direction_change.mean()[0]
        new_row_data['direction_change__median'] = direction_change.quantile(.5)[0]
        new_row_data['direction_change__first_quantile'] = direction_change.quantile(.25)[0]
        new_row_data['direction_change__last_quantile'] = speed.quantile(.75)[0]

        # proximity_to_crosswalk
        new_row_data['proximity_to_crosswalk__start'] = proximity_to_crosswalk

        # proximity_to_road
        new_row_data['proximity_to_road__start'] = proximity_to_road_start
        new_row_data['proximity_to_road__min'] = proximity_to_road_min

        # proximity_to_intersection
        new_row_data['proximity_to_inter__start'] = proximity_to_inter_start
        new_row_data['proximity_to_inter__min'] = proximity_to_inter_min

        # Action Features
        # TODO add last point
        new_row_data['crossing'] = crossing_feature
        new_row_data['illegal_crossing'] = illegal_crossing_feature
        new_row_data['turning'] = turn_feature

        # Starting Point Feature
        # TODO add the count of points in each location type, 0 if not in a location type
        new_row_data['starting_point'] = check_point_location(Point(x[0], y[0]))

        new_row_df = pd.DataFrame([new_row_data])
        df = pd.concat([df, new_row_df], ignore_index=True)

    return df

if __name__ == "__main__":
    sind = SinD()
    map = sind.map
    input_len = 90
    # TODO Keep 90 e.g steps (experiment with that number) without overlaping and keep the last value as a feature of all the variables (e.g direction, speed, acc etc.) 
    # ADD also heading
    # dataset, data_type = sind.data(input_len=input_len), ''
    dataset, data_type = sind.pedestrian_data, '__full'

    df = create_dataframe(dataset, input_len)
    df.to_csv(ROOT + f"/dataset_created{data_type}.csv", encoding='utf-8', index=False)