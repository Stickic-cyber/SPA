
from keypoint_detector import detect_keypoints_from_video
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import iqr

def calculate_upper_stability(landmarks):
    """
    计算胸部中点水平方向的稳定性，并绘制箱线图，统计上下四分位距范围内的占比。
    
    参数：
    landmarks: 3D关键点数据，每一帧包含一个landmark的列表，返回值为3D坐标(x, y, z, visibility)。
    
    返回：
    None（绘制图形并输出稳定性评分）。
    """
    # 提取胸部中点水平方向的x轴数据
    x_coords = []
    for frame_landmarks in landmarks:
        if frame_landmarks is not None:
            # 获取胸部中心点坐标（通过左肩和右肩的平均位置来估算）
            left_shoulder = frame_landmarks[11]  # 左肩
            right_shoulder = frame_landmarks[12]  # 右肩

            # 计算胸部中间点的x坐标
            chest_x = (left_shoulder.x + right_shoulder.x) / 2
            x_coords.append(chest_x)

    if len(x_coords) == 0:
        print("没有有效的胸部中点坐标数据，无法绘制图表。")
        return

    # 计算四分位数
    Q1 = np.percentile(x_coords, 25)
    Q3 = np.percentile(x_coords, 75)
    IQR_value = Q3 - Q1
    median_x = np.median(x_coords)

    # 计算上下四分位距范围内的数据占比
    in_iqr = [x for x in x_coords if Q1 <= x <= Q3]
    in_iqr_percentage = len(in_iqr) / len(x_coords) * 100

    # 绘制箱线图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(figsize=(10, 6))
    plt.boxplot(x_coords, vert=False, patch_artist=True, 
                boxprops=dict(facecolor="skyblue", color="black"), 
                whiskerprops=dict(color="black"), 
                flierprops=dict(marker='o', color='red', markersize=5))
    
    plt.title("胸部中点 X 坐标箱线图")
    plt.xlabel("胸部中点 X 坐标")
    plt.grid(True)
    plt.show()
    
    return Q1, median_x, Q3, in_iqr_percentage


    # 输出四分位数和占比
    print(f"胸部中点 X 坐标的四分位数：")
    print(f"Q1（下四分位数）：{Q1:.4f}")
    print(f"Q3（上四分位数）：{Q3:.4f}")
    print(f"中位数：{median_x:.4f}")
    print(f"上下四分位距范围内的数据占比：{in_iqr_percentage:.2f}%")


if __name__ == '__main__':
    # 调用示例
    video_path = r'dataset\input\shoot_with_rim_2.mp4'
    all_landmarks, (w, h, fps) = detect_keypoints_from_video(video_path)
    print(calculate_upper_stability(all_landmarks))
