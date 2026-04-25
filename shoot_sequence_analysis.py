import numpy as np
import pandas as pd
from typing import List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 关节和关键点索引的映射
# Mediapipe 关键点索引参考: https://google.github.io/mediapipe/solutions/pose.html
KEYPOINT_MAP = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_foot_index': 31, 'right_foot_index': 32,
}

# 理想发力顺序，用于计算逆序对 (从下往上)
IDEAL_SEQUENCE = ['ankle', 'knee', 'hip', 'shoulder', 'elbow']
MAX_INVERSIONS = len(IDEAL_SEQUENCE) * (len(IDEAL_SEQUENCE) - 1) / 2 # 5个关节，最大逆序对数 = 5*4/2 = 10

def get_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算由三个点 P1, P2, P3 构成的夹角，顶点为 P2。"""
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def get_closest_side(first_frame_kpts_3d: np.ndarray) -> str:
    """
    根据肩部和髋部的x坐标，判断哪一侧身体更靠近摄像头。
    此函数直接处理 NumPy 数组格式的关键点数据。
    """
    if first_frame_kpts_3d is None or len(first_frame_kpts_3d) == 0:
        return 'right' # 默认选择右侧，如果数据无效

    left_shoulder_x = first_frame_kpts_3d[KEYPOINT_MAP['left_shoulder']][0]
    right_shoulder_x = first_frame_kpts_3d[KEYPOINT_MAP['right_shoulder']][0]
    left_hip_x = first_frame_kpts_3d[KEYPOINT_MAP['left_hip']][0]
    right_hip_x = first_frame_kpts_3d[KEYPOINT_MAP['right_hip']][0]

    left_x_range = abs(left_shoulder_x - left_hip_x)
    right_x_range = abs(right_shoulder_x - right_hip_x)

    # 简单地通过肩部和髋部的x坐标差值判断，x坐标差值越小，说明该侧更靠近摄像头
    if left_x_range < right_x_range:
        return 'left'
    else:
        return 'right'
        
from scipy.signal import savgol_filter

def smooth_curve(data: np.ndarray, window_size: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    对数据进行平滑处理，使用 Savitzky-Golay 滤波器。

    参数:
    data (np.ndarray): 输入的一维数据数组。
    window_size (int): 滤波器的窗口大小。为了获得最佳结果，该值通常为奇数。
                       如果输入偶数，函数会将其自动加1转为奇数。
    polyorder (int): 用于拟合样本的多项式阶数。必须小于 window_size。
                     阶数越高，对原始数据的拟合越好，但平滑效果越差。
                     常用的值为2或3。

    返回:
    np.ndarray: 平滑后的一维数据数组。
    """
    # 1. 检查数据长度，如果数据太少则无法滤波
    if len(data) < window_size:
        print(f"警告: 数据长度 ({len(data)}) 小于窗口大小 ({window_size})，无法进行滤波。返回原始数据。")
        return data

    # 2. 确保 window_size 是一个正奇数
    if window_size <= 0:
        raise ValueError("window_size 必须是正数。")
    if window_size % 2 == 0:
        window_size += 1
        print(f"信息: window_size 已被调整为奇数 {window_size}。")

    # 3. 确保 polyorder 小于 window_size
    if polyorder >= window_size:
        raise ValueError("polyorder 必须小于 window_size。")

    # 4. 应用 Savitzky-Golay 滤波器
    # mode='interp' 会对边缘进行插值，效果通常比默认的 'mirror' 更好
    smoothed_data = savgol_filter(data, window_size, polyorder, mode='interp')
    
    return smoothed_data

def calculate_acceleration(angles: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """
    计算角度的二阶差分近似为角加速度，并进行平滑。
    假设帧率恒定，时间间隔为 1。
    加速度的计算：a(t) ≈ (v(t+1) - v(t-1)) / (2*dt) 或 a(t) ≈ (x(t+1) - 2*x(t) + x(t-1)) / dt^2
    这里使用二阶差分: a_i = x_{i+1} - 2x_i + x_{i-1}
    """
    if len(angles) < 3:
        return np.zeros_like(angles)
    
    # 计算二阶差分
    acceleration = np.diff(angles, n=2)
    # 对结果进行填充，以匹配原始数组长度（首尾各填充一个 0）
    acceleration = np.concatenate(([0], acceleration, [0]))
    
    # 对加速度进行平滑处理
    smoothed_acceleration = smooth_curve(acceleration, window_size=smooth_window)
    return smoothed_acceleration

def evaluate_power_sequence(
    pose_landmarks_3d: List[np.ndarray],
    frame_rate: int = 30,
    plot_output_path: str = 'joint_angles_and_acceleration_plot.png',
) -> float:
    """
    分析投篮的发力顺序并计算基于逆序对的评分。
    发力点判断逻辑已优化：先计算各关节的角加速度（二阶差分），再根据加速度最大的点设为发力点。
    同时，绘制五个关节的角度变化图和发力加速度变化图。
    
    参数:
    pose_landmarks_3d: 包含每一帧所有3D关键点坐标的列表，每个元素是一个np.ndarray。
    frame_rate: 视频的帧率（用于时间轴参考）。
    
    返回:
    power_sequence_score: 介于 0 到 100 的评分。
    """
    if not pose_landmarks_3d or pose_landmarks_3d[0] is None:
        print("姿态关键点数据无效，无法计算发力顺序得分。")
        return 60.0

    # 1. 判断靠近摄像头的一侧
    side = get_closest_side(pose_landmarks_3d[0])
    print(f"检测到靠近摄像头的一侧是: {side}")
    
    # 2. 计算每个关节在每一帧的角度
    angles_series = {joint: [] for joint in IDEAL_SEQUENCE}

    side_prefix = 'left' if side == 'left' else 'right'

    for frame_kpts in pose_landmarks_3d:
        if frame_kpts is None:
            continue

        # 使用 3D 坐标计算角度
        try:
            ankle = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_ankle']]
            knee = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_knee']]
            hip = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_hip']]
            shoulder = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_shoulder']]
            elbow = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_elbow']]
            wrist = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_wrist']]
            foot_index = frame_kpts[KEYPOINT_MAP[f'{side_prefix}_foot_index']]

            # 关节角度计算
            # 膝盖角度: 髋-膝-踝
            angles_series['knee'].append(get_angle(hip, knee, ankle))
            # 髋关节角度: 肩-髋-膝
            angles_series['hip'].append(get_angle(shoulder, hip, knee))
            # 肩关节角度: 肘-肩-髋
            angles_series['shoulder'].append(get_angle(elbow, shoulder, hip))
            # 肘关节角度: 肩-肘-腕（使用假想点或真实点）
            p_wrist = wrist if wrist[2] != 0 else elbow + np.array([0, -1, 0])
            angles_series['elbow'].append(get_angle(shoulder, elbow, p_wrist))
            # 踝关节角度: 膝-踝-脚趾（使用假想点或真实点）
            p_foot_index = foot_index if foot_index[2] != 0 else ankle + np.array([0, -1, 0])
            angles_series['ankle'].append(get_angle(knee, ankle, p_foot_index))

        except IndexError:
            # 某些帧可能缺少关键点，跳过
            continue

    # 3. 对角度数据进行平滑处理
    # 4. 计算角加速度
    acceleration_series = {}
    for joint, angles in angles_series.items():
        if angles:
            angles_arr = np.array(angles)
            # 先平滑角度
            angles_series[joint] = smooth_curve(angles_arr, window_size=5).tolist()
            # 再计算加速度
            # 使用平滑后的角度计算加速度
            acceleration_series[joint] = calculate_acceleration(np.array(angles_series[joint]), smooth_window=5)
        else:
            acceleration_series[joint] = np.array([])
            
    # 5. 找到肩关节的最高点（角度最大点）作为分析截止时间点
    if not angles_series['shoulder']:
        print("肩关节数据无效，无法计算发力顺序得分。")
        return 60.0
        
    shoulder_max_angle_index = np.argmax(angles_series['shoulder'])
    print(f"检测到肩关节最高点（角度）在帧数: {shoulder_max_angle_index}")

    # 6. 找到截止点之前每个关节的**最大加速度点** (发力点)
    power_points = {}
    for joint, accels in acceleration_series.items():
        if len(accels) > 0:
            # 找到截止点之前加速度（绝对值，通常是正值）最大值的索引
            # 注意：投篮发力通常是伸展，对应角度变大，二阶导数在屈伸转折点后可能为正的最大值。
            # 这里找正向加速的最大值点
            
            # 截取到截止点
            accel_slice = accels[:shoulder_max_angle_index + 1]
            if len(accel_slice) > 0:
                # 寻找正向最大加速度点
                max_accel_index = np.argmax(accel_slice)
                power_points[joint] = max_accel_index
            else:
                # 如果切片为空，则设为第一个点（作为默认）
                power_points[joint] = 0 
    
    if len(power_points) < len(IDEAL_SEQUENCE):
        print("无法检测所有发力点，发力顺序得分设置为60分。")
        return 60.0

    # 7. 绘制角度变化图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.style.use('ggplot')
    
    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'p']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # --- 角度变化图 (ax1) ---
    for i, (joint, angles) in enumerate(angles_series.items()):
        min_angle_index = power_points.get(joint)
        frames = np.arange(len(angles))
        
        # 绘制截止点前的曲线
        ax1.plot(frames[:shoulder_max_angle_index + 1], 
                 angles[:shoulder_max_angle_index + 1], 
                 label=joint, color=colors[i], linewidth=2)
        
        # 绘制截止点后的曲线（灰色蒙版）
        if shoulder_max_angle_index < len(angles) - 1:
            ax1.plot(frames[shoulder_max_angle_index:], 
                      angles[shoulder_max_angle_index:], 
                      color='gray', linestyle='--', linewidth=1)
        
        # 标记发力点（加速度最大点）
        if min_angle_index is not None and min_angle_index < len(angles):
            # 在角度图上标记加速度最大的点
            ax1.plot(min_angle_index, angles[min_angle_index], 
                     marker='*', markersize=12, 
                     color=colors[i], label=f'{joint} 发力点', linestyle='None')
            ax1.text(min_angle_index, angles[min_angle_index] + 5, 
                     f'({min_angle_index})', fontsize=10, ha='center', va='bottom', color=colors[i])

    ax1.axvline(shoulder_max_angle_index, color='r', linestyle='-.', linewidth=1, label='肩部最大角度截止点')
    ax1.set_title('关节角度随时间变化曲线 (发力点基于最大加速度)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('角度 (°)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True)


    # --- 加速度变化图 (ax2) ---
    for i, (joint, accels) in enumerate(acceleration_series.items()):
        max_accel_index = power_points.get(joint)
        frames = np.arange(len(accels))
        
        # 绘制截止点前的曲线
        ax2.plot(frames[:shoulder_max_angle_index + 1], 
                 accels[:shoulder_max_angle_index + 1], 
                 label=joint + '加速度', color=colors[i], linewidth=2)
        
        # 绘制截止点后的曲线（灰色蒙版）
        if shoulder_max_angle_index < len(accels) - 1:
            ax2.plot(frames[shoulder_max_angle_index:], 
                      accels[shoulder_max_angle_index:], 
                      color='gray', linestyle='--', linewidth=1)
        
        # 标记最大加速度点
        if max_accel_index is not None and max_accel_index < len(accels):
            ax2.plot(max_accel_index, accels[max_accel_index], 
                     marker=markers[i], markersize=10, 
                     color=colors[i], label=f'{joint} 最大加速度点', linestyle='None')
            # 标记数值
            ax2.text(max_accel_index, accels[max_accel_index] * 1.05, 
                     f'{accels[max_accel_index]:.1f}', fontsize=10, ha='center', va='bottom', color=colors[i])

    ax2.axvline(shoulder_max_angle_index, color='r', linestyle='-.', linewidth=1, label='肩部最大角度截止点')
    ax2.axhline(0, color='k', linestyle=':', linewidth=0.8) # 零线
    ax2.set_title('关节角加速度随时间变化曲线', fontsize=16, fontweight='bold')
    ax2.set_xlabel('帧数', fontsize=12)
    ax2.set_ylabel('角加速度 (deg/frame²)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close(fig)
    print(f"关节角度和加速度变化图已保存为 {plot_output_path}")

    # 8. 确定实际发力顺序 (基于加速度最大点的帧数)
    actual_sequence_sorted = sorted(power_points.items(), key=lambda item: item[1])
    actual_sequence = [item[0] for item in actual_sequence_sorted]
    
    print(f"检测到的发力顺序 (基于最大加速度): {actual_sequence}")

    # 9. 计算逆序对数量
    inversions = 0
    for i in range(len(IDEAL_SEQUENCE)):
        for j in range(i + 1, len(IDEAL_SEQUENCE)):
            ideal_joint_1 = IDEAL_SEQUENCE[i]
            ideal_joint_2 = IDEAL_SEQUENCE[j]
            
            # 找到这两个关节在实际序列中的位置
            # 确保关节都在实际序列中
            if ideal_joint_1 in actual_sequence and ideal_joint_2 in actual_sequence:
                pos1 = actual_sequence.index(ideal_joint_1)
                pos2 = actual_sequence.index(ideal_joint_2)
                
                # 如果在实际序列中的位置反了，则是一个逆序对 (期望 pos1 < pos2)
                if pos1 > pos2:
                    inversions += 1
            # else: 理论上前面检查了 len(power_points)，应该不会发生

    print(f"逆序对数量: {inversions}")
    
    # 10. 计算评分
    if MAX_INVERSIONS > 0:
        score = 100 - (inversions / MAX_INVERSIONS) * 40
    else:
        score = 100.0 # 避免除以零
        
    return score

