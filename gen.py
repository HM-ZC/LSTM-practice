import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 常量
g = 9.81  # 重力加速度 m/s^2
basket_height = 3.05  # 篮筐高度，单位：米
basket_distance = 7.5  # 篮筐距离投篮点的水平距离，单位：米
three_point_radius = 7.24  # 三分线半径，单位：米

# 随机生成从三分线某点投篮的初始位置
def get_random_three_point_position():
    theta = np.random.uniform(-np.pi / 3, np.pi / 3)  # 限制角度在 [-60°, 60°] 范围，避免过大偏移
    x_pos = basket_distance - three_point_radius * np.cos(theta)  # 水平距离
    z_pos = three_point_radius * np.sin(theta)  # 左右偏移
    return x_pos, z_pos

# 计算使投篮的速度和角度
def get_speed_and_angle(start_x, start_z):
    launch_angle = np.radians(np.random.uniform(35, 45))  # 随机生成一个合理的出射角度
    initial_speed = np.random.uniform(8, 11)  # 速度适中
    return initial_speed, launch_angle

# 计算初速度的分解
def calculate_velocity_components(start_x, start_z, initial_speed, launch_angle):
    distance_to_basket = np.sqrt((basket_distance - start_x) ** 2 + start_z ** 2)
    vx = initial_speed * np.cos(launch_angle) * (basket_distance - start_x) / distance_to_basket
    vz = initial_speed * np.cos(launch_angle) * (-start_z) / distance_to_basket
    vy = initial_speed * np.sin(launch_angle)
    return vx, vy, vz

# 模拟单次投篮并生成轨迹，包括 z 方向上的偏移
def generate_shot_trajectory(shot_id, initial_speed, launch_angle, initial_height, start_x, start_z):
    vx, vy, vz = calculate_velocity_components(start_x, start_z, initial_speed, launch_angle)

    time_interval = 0.05  # 时间间隔，单位：秒
    total_time = 3.0  # 总时间，单位：秒
    time_steps = np.arange(0, total_time + time_interval, time_interval)
    time_steps = np.round(time_steps, 2)  # 四舍五入时间步，确保精度

    # 初始化轨迹数据
    trajectory_data = {
        'shot_id': [],
        'timestamp': [],
        'x': [],
        'y': [],
        'z': [],
        'vx': [],
        'vy': [],
        'vz': []
    }

    for t in time_steps:
        x_pos = start_x + vx * t
        y_pos = initial_height + vy * t - 0.5 * g * t ** 2
        z_pos = start_z + vz * t

        # 如果篮球落地或超过篮筐区域（出界），停止生成
        if y_pos < 0 or x_pos > basket_distance + 1 or abs(z_pos) > three_point_radius:
            break

        # 保存轨迹数据
        trajectory_data['shot_id'].append(shot_id)
        trajectory_data['timestamp'].append(t)
        trajectory_data['x'].append(x_pos)
        trajectory_data['y'].append(y_pos)
        trajectory_data['z'].append(z_pos)
        trajectory_data['vx'].append(vx)
        trajectory_data['vy'].append(vy - g * t)
        trajectory_data['vz'].append(vz)

    return pd.DataFrame(trajectory_data)

# 生成投篮数据并保存为CSV
def batch_generate_shots(num_shots, output_csv):
    all_shots_data = []  # 用于存储所有投篮的数据

    for shot_id in range(num_shots):
        start_x, start_z = get_random_three_point_position()
        initial_speed, launch_angle = get_speed_and_angle(start_x, start_z)
        initial_height = np.random.uniform(1.8, 2.2)  # 出手高度在 1.8 到 2.2 米之间

        # 生成单个投篮的轨迹
        shot_data = generate_shot_trajectory(shot_id, initial_speed, launch_angle, initial_height, start_x, start_z)
        all_shots_data.append(shot_data)

    # 合并所有投篮数据并保存为一个CSV文件
    all_shots_df = pd.concat(all_shots_data, ignore_index=True)
    all_shots_df.to_csv(output_csv, index=False)
    print(f"All shots data saved to: {output_csv}")

# 生成10次投篮，并将所有结果保存到一个CSV文件中
batch_generate_shots(1000, 'all_shots_data.csv')