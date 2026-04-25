from keypoint_detector import detect_keypoints_from_video
from view_calculator import compute_eye_view_lines
from yolo_detector import detect_on_video_yolov8

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ---- 工具函数 ----
def calculate_angle(a, b, c):
    """
    计算三点(a,b,c) 构成关节角度 (单位: °)。
    a,b,c 是 mediapipe 的 landmark，包含 x,y,z。
    """
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])  # 关节点
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    # 夹角 (弧度)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # 转换为度数
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def compute_angle_sequence(all_landmarks, joint_name="RIGHT_ELBOW"):
    """
    从关键点序列提取某个关节的角度时间序列 (单位: °)。
    例如 joint_name="RIGHT_ELBOW" => 右肘角度（肩-肘-腕）
    """
    mp_pose = __import__("mediapipe").solutions.pose

    joint_map = {
        "RIGHT_ELBOW": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                        mp_pose.PoseLandmark.RIGHT_WRIST),
        "LEFT_ELBOW": (mp_pose.PoseLandmark.LEFT_SHOULDER,
                       mp_pose.PoseLandmark.LEFT_ELBOW,
                       mp_pose.PoseLandmark.LEFT_WRIST),
        "RIGHT_KNEE": (mp_pose.PoseLandmark.RIGHT_HIP,
                       mp_pose.PoseLandmark.RIGHT_KNEE,
                       mp_pose.PoseLandmark.RIGHT_ANKLE),
        "LEFT_KNEE": (mp_pose.PoseLandmark.LEFT_HIP,
                      mp_pose.PoseLandmark.LEFT_KNEE,
                      mp_pose.PoseLandmark.LEFT_ANKLE),
    }

    if joint_name not in joint_map:
        raise ValueError(f"Unsupported joint_name {joint_name}")

    a_idx, b_idx, c_idx = joint_map[joint_name]
    angle_seq = []

    for lm in all_landmarks:
        if lm is None:
            angle_seq.append(None)
            continue
        angle = calculate_angle(lm[a_idx], lm[b_idx], lm[c_idx])
        angle_seq.append(angle)

    return np.array([a for a in angle_seq if a is not None])


def compute_jerk_rms(angle_seq, fps):
    """
    输入角度序列 (单位: °) 和采样率 fps，计算 jerk RMS。
    输出 jerk RMS 的单位为 °/s^3。
    """
    if len(angle_seq) < 5:
        return None

    # 一阶、二阶、三阶导数（中心差分）
    dt = 1.0 / fps
    vel = np.gradient(angle_seq, dt)       # 角速度 (°/s)
    acc = np.gradient(vel, dt)             # 角加速度 (°/s²)
    jerk = np.gradient(acc, dt)            # 角 jerk (°/s³)

    # jerk RMS
    rms = np.sqrt(np.mean(jerk**2))
    return rms, jerk



def plot_angle_dynamics(angle_seq, fps, joint_name="Joint"):
    """
    输入角度序列和fps，绘制角度 / 角速度 / 角加速度 / jerk 曲线
    """
    dt = 1.0 / fps
    t = np.arange(len(angle_seq)) * dt

    vel = np.gradient(angle_seq, dt)       # 角速度
    acc = np.gradient(vel, dt)             # 角加速度
    jerk = np.gradient(acc, dt)            # 角jerk

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, angle_seq, label="Angle (°)")
    axs[0].set_ylabel("Angle (°)")
    axs[0].legend()

    axs[1].plot(t, vel, label="Angular Velocity")
    axs[1].set_ylabel("Velocity (°/s)")
    axs[1].legend()

    axs[2].plot(t, acc, label="Angular Acceleration")
    axs[2].set_ylabel("Acceleration (°/s²)")
    axs[2].legend()

    axs[3].plot(t, jerk, label="Angular Jerk")
    axs[3].set_ylabel("Jerk (°/s³)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend()

    fig.suptitle(f"{joint_name} Dynamics", fontsize=14)
    plt.tight_layout()
    plt.show()

    return vel, acc, jerk

# 1. 提取关键点
all_landmarks, (w, h, fps) = detect_keypoints_from_video(r'dataset\input\shoot_with_rim_2.mp4')

def smooth_angle_sequence(angle_seq, window_length=7, polyorder=2):
    """
    对角度序列进行平滑处理
    window_length: 滑动窗口长度，必须为奇数
    polyorder: 多项式阶数
    """
    angle_seq = np.asarray(angle_seq)
    n = len(angle_seq)
    if n < 5:
        return angle_seq
    # 保证窗口长度不超过序列长度且为奇数
    wl = min(window_length, n if n % 2 == 1 else n-1)
    if wl < 3:
        return angle_seq
    return savgol_filter(angle_seq, window_length=wl, polyorder=polyorder)

# ---------------- 使用平滑后的序列 ----------------
# 2. 计算右肘角度序列
angles = compute_angle_sequence(all_landmarks, joint_name="RIGHT_ELBOW")

# 平滑处理
angles_smooth = smooth_angle_sequence(angles, window_length=7, polyorder=2)

# 3. 计算 jerk RMS（使用平滑序列）
jerk_rms, jerk_seq = compute_jerk_rms(angles_smooth, fps)
print("右肘 jerk RMS (平滑后) =", jerk_rms)

# 绘制动力学曲线
vel, acc, jerk = plot_angle_dynamics(angles_smooth, fps, joint_name="Right Elbow (Smoothed)")
