import matplotlib
import numpy as np
import pandas as pd
from keypoint_detector import detect_keypoints_from_video
import mediapipe as mp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
mp_pose = mp.solutions.pose

# ---------- 工具函数 ----------
def calculate_angle(a, b, c):
    """
    计算三点 a-b-c 在 b 点的夹角，返回角度（0~180°）
    a, b, c: [x, y, z]
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    # 归一化
    ba_norm = ba / (np.linalg.norm(ba) + 1e-6)
    bc_norm = bc / (np.linalg.norm(bc) + 1e-6)

    cosine = np.dot(ba_norm, bc_norm)
    cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性
    angle = np.degrees(np.arccos(cosine))
    return angle

def extract_joint_angles(all_landmarks, whf):
    """
    输入:
        all_landmarks: detect_keypoints_from_video 的输出 (list，每帧关键点列表)
        whf: (w, h, fps)
    输出:
        angles_dict: { "left_knee": [...], "right_knee": [...], ... }
    """
    angle_names = ["left_knee", "right_knee",
                   "left_hip", "right_hip",
                   "left_shoulder", "right_shoulder",
                   "left_elbow", "right_elbow",
                   "left_wrist", "right_wrist"]

    # 存储结果
    angles_dict = {name: [] for name in angle_names}

    for landmarks in all_landmarks:
        if landmarks is None:
            # 如果当前帧没检测到，填 None
            for k in angles_dict:
                angles_dict[k].append(None)
            continue

        lm = landmarks  # MediaPipe pose_landmarks.landmark

        # 按照人体结构定义角度
        try:
            # 膝盖: hip - knee - ankle
            left_knee = calculate_angle(
                [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y, lm[mp_pose.PoseLandmark.LEFT_HIP].z],
                [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y, lm[mp_pose.PoseLandmark.LEFT_KNEE].z],
                [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y, lm[mp_pose.PoseLandmark.LEFT_ANKLE].z]
            )
            right_knee = calculate_angle(
                [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y, lm[mp_pose.PoseLandmark.RIGHT_HIP].z],
                [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y, lm[mp_pose.PoseLandmark.RIGHT_KNEE].z],
                [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].z]
            )

            # 髋: shoulder - hip - knee
            left_hip = calculate_angle(
                [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z],
                [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y, lm[mp_pose.PoseLandmark.LEFT_HIP].z],
                [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y, lm[mp_pose.PoseLandmark.LEFT_KNEE].z]
            )
            right_hip = calculate_angle(
                [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z],
                [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y, lm[mp_pose.PoseLandmark.RIGHT_HIP].z],
                [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y, lm[mp_pose.PoseLandmark.RIGHT_KNEE].z]
            )

            # 肩: elbow - shoulder - hip
            left_shoulder = calculate_angle(
                [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y, lm[mp_pose.PoseLandmark.LEFT_ELBOW].z],
                [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z],
                [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y, lm[mp_pose.PoseLandmark.LEFT_HIP].z]
            )
            right_shoulder = calculate_angle(
                [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z],
                [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z],
                [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y, lm[mp_pose.PoseLandmark.RIGHT_HIP].z]
            )

            # 肘: shoulder - elbow - wrist
            left_elbow = calculate_angle(
                [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z],
                [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y, lm[mp_pose.PoseLandmark.LEFT_ELBOW].z],
                [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y, lm[mp_pose.PoseLandmark.LEFT_WRIST].z]
            )
            right_elbow = calculate_angle(
                [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z],
                [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z],
                [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y, lm[mp_pose.PoseLandmark.RIGHT_WRIST].z]
            )

            # 腕: elbow - wrist - index
            left_wrist = calculate_angle(
                [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y, lm[mp_pose.PoseLandmark.LEFT_ELBOW].z],
                [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y, lm[mp_pose.PoseLandmark.LEFT_WRIST].z],
                [lm[mp_pose.PoseLandmark.LEFT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_INDEX].y, lm[mp_pose.PoseLandmark.LEFT_INDEX].z]
            )
            right_wrist = calculate_angle(
                [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z],
                [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y, lm[mp_pose.PoseLandmark.RIGHT_WRIST].z],
                [lm[mp_pose.PoseLandmark.RIGHT_INDEX].x, lm[mp_pose.PoseLandmark.RIGHT_INDEX].y, lm[mp_pose.PoseLandmark.RIGHT_INDEX].z]
            )

        except Exception:
            left_knee = right_knee = left_hip = right_hip = 0
            left_shoulder = right_shoulder = left_elbow = right_elbow = 0
            left_wrist = right_wrist = 0

        # 保存到字典
        angles_dict["left_knee"].append(left_knee)
        angles_dict["right_knee"].append(right_knee)
        angles_dict["left_hip"].append(left_hip)
        angles_dict["right_hip"].append(right_hip)
        angles_dict["left_shoulder"].append(left_shoulder)
        angles_dict["right_shoulder"].append(right_shoulder)
        angles_dict["left_elbow"].append(left_elbow)
        angles_dict["right_elbow"].append(right_elbow)
        angles_dict["left_wrist"].append(left_wrist)
        angles_dict["right_wrist"].append(right_wrist)

    return angles_dict

def plot_joint_angles(angles_dict, fps):
    """
    可视化10个关节角度曲线
    """
    t = np.arange(len(next(iter(angles_dict.values())))) / fps
    plt.figure(figsize=(16, 12))
    for i, (name, values) in enumerate(angles_dict.items(), 1):
        plt.subplot(5, 2, i)
        plt.plot(t, values, label=name)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(name)
        plt.legend()
    plt.tight_layout()
    plt.show()


def save_joint_angle_plot(angles_dict, fps, output_path, title):
    """
    保存关节角度变化图。
    """
    if not angles_dict:
        return

    t = np.arange(len(next(iter(angles_dict.values())))) / fps
    fig = plt.figure(figsize=(16, 12))
    for i, (name, values) in enumerate(angles_dict.items(), 1):
        ax = fig.add_subplot(5, 2, i)
        series = np.array(values, dtype=np.float64)
        ax.plot(t, series, label=name, linewidth=1.6, color="#0f766e")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


from scipy.signal import savgol_filter

def process_angles(angles_dict, window_length=15, polyorder=3):
    """
    处理关节角度数据：异常值检测和插值，然后平滑滤波。

    参数:
        angles_dict (dict): 原始关节角度数据字典。
        window_length (int): Savitzky-Golay 滤波器的窗口长度，必须是正奇数。
        polyorder (int): 拟合多项式的阶数。

    返回:
        filtered_angles_dict (dict): 处理后的关节角度数据字典。
    """
    filtered_angles_dict = {}

    for name, angles in angles_dict.items():
        # 1. IQR异常值处理
        angles_np = np.array(angles, dtype=np.float64)  # 转换为 NumPy 数组
        
        # 移除 None 值以进行 IQR 计算
        valid_angles = angles_np[~np.isnan(angles_np)]

        if len(valid_angles) == 0:
            filtered_angles_dict[name] = angles
            continue

        q1, q3 = np.percentile(valid_angles, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 标记异常值并进行插值
        processed_angles = angles_np.copy()
        for i in range(len(processed_angles)):
            # 处理 NaN 值（来自 None）
            if np.isnan(processed_angles[i]):
                processed_angles[i] = np.nan
                continue

            # 处理 IQR 异常值
            if processed_angles[i] < lower_bound or processed_angles[i] > upper_bound:
                processed_angles[i] = np.nan
        
        # 插值处理 NaN 值
        processed_angles = pd.Series(processed_angles).interpolate(limit_direction='both').values
        
        # 2. Savitzky-Golay 滤波
        # 窗口长度不能超过数据点数量
        win_len = min(window_length, len(processed_angles) - 1)
        if win_len % 2 == 0:
            win_len += 1  # 确保是奇数
        
        # 如果数据太短，跳过滤波
        if win_len < polyorder + 1:
            filtered_angles = processed_angles
        else:
            filtered_angles = savgol_filter(processed_angles, window_length=win_len, polyorder=polyorder)
            
        filtered_angles_dict[name] = filtered_angles.tolist()

    return filtered_angles_dict

def calculate_angular_velocity(angles_dict, fps):
    """
    计算关节的角速度（角度变化率）。

    参数:
        angles_dict (dict): 经过处理（平滑后）的关节角度数据。
        fps (int): 视频帧率。

    返回:
        angular_velocity_dict (dict): 每个关节的角速度数据。
    """
    angular_velocity_dict = {}
    for name, angles in angles_dict.items():
        # 使用 np.gradient 计算一阶导数，即角速度
        velocity = np.gradient(angles, 1/fps)
        angular_velocity_dict[name] = velocity.tolist()
    return angular_velocity_dict

def analyze_power_sequence(angular_velocity_dict, fps):
    """
    分析发力顺序。

    参数:
        angular_velocity_dict (dict): 关节角速度数据。
        fps (int): 视频帧率。

    返回:
        sequence_list (list): 按发力时间排序的元组列表 [(关节名, 峰值时间), ...]。
    """
    peak_times = []
    for name, velocities in angular_velocity_dict.items():
        # 排除初始或结束时可能出现的异常值
        if len(velocities) < 5:
            continue
            
        # 找到角速度绝对值的最大值索引
        peak_idx = np.argmax(np.abs(velocities))
        
        # 计算峰值时间
        peak_time = peak_idx / fps
        peak_times.append((name, peak_time))
        
    # 按峰值时间排序
    sequence_list = sorted(peak_times, key=lambda x: x[1])
    return sequence_list


if __name__ == '__main__':
    import pandas as pd
    
    video_path = r'dataset\input\curry_org.mp4'
    all_landmarks, (w, h, fps) = detect_keypoints_from_video(video_path)
    
    # 1. 提取和处理角度数据
    angles_dict = extract_joint_angles(all_landmarks, (w, h, fps))
    filtered_angles_dict = process_angles(angles_dict)
    
    # 2. 计算角速度
    angular_velocity_dict = calculate_angular_velocity(filtered_angles_dict, fps)
    
    # 3. 分析发力顺序
    power_sequence = analyze_power_sequence(angular_velocity_dict, fps)
    
    print("\n评估发力顺序:")
    for name, time in power_sequence:
        print(f"{name}: 峰值发力时间为 {time:.2f} 秒")
        
    # 可选：可视化角速度曲线
    plt.figure(figsize=(16, 12))
    for i, (name, values) in enumerate(angular_velocity_dict.items(), 1):
        plt.subplot(5, 2, i)
        plt.plot(np.arange(len(values)) / fps, values, label=name)
        plt.title(f'{name} Angular Velocity')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (deg/s)")
        plt.legend()
    plt.tight_layout()
    plt.show()
