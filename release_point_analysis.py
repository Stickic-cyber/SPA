import numpy as np
import pandas as pd
from typing import List, Any, Tuple

# 为了方便，我们在这里定义MediaPipe关键点的索引
# 可以在MEDIAPIPE_POSE_KEYPOINTS列表中找到对应的索引值
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_EAR = 8

def analyze_shooting_form(pose_landmarks: List[Any], yolo_df: pd.DataFrame, video_info: Tuple[int, int, int]) -> float:
    """
    分析投篮姿势并根据特定规则进行评分。

    评分规则:
    1. 首先判断每一帧中，人体的右侧大臂（从右肩到右肘的向量）是否在水平方向的±15°范围内。
    2. 如果整个视频中都没有任何一帧满足此条件，则直接返回60分。
    3. 如果至少有一帧满足条件，则基础分为60分，并进入下一步计算。
    4. 在所有满足“大臂水平”条件的帧中，计算篮球的识别框中心点位于“以右耳为原点的第一象限”内的帧数占比。
    5. 最终得分 = 60 + (该占比 * 40)。

    Args:
        pose_landmarks (List[Any]): 来自 keypoint_detector.py 的姿态关键点列表。
        yolo_df (pd.DataFrame): 来自 yolo_detector.py 的目标检测结果数据。
        video_info (Tuple[int, int, int]): 包含视频宽度、高度和帧率的元组 (w, h, fps)。

    Returns:
        float: 计算出的综合得分。
    """
    w, h, _ = video_info
    
    # 步骤 1: 预处理YOLO数据，筛选出篮球并计算中心点
    df_ball = yolo_df[yolo_df['class'] == 'ball'].copy()
    if df_ball.empty:
        print("分析警告：未在视频中检测到'ball'，无法进行出手姿势分析。默认返回基础分60分。")
        return 60.0
        
    # 计算篮球中心点坐标
    df_ball['center_x'] = (df_ball['x_min'] + df_ball['x_max']) / 2
    df_ball['center_y'] = (df_ball['y_min'] + df_ball['y_max']) / 2
    # 将处理后的篮球数据按帧号存入字典，以提高查找效率
    ball_by_frame = {
        frame: group[['center_x', 'center_y']].iloc[0] 
        for frame, group in df_ball.groupby('frame')
    }

    # 步骤 2: 遍历所有帧，找出右大臂符合角度要求的帧
    valid_arm_angle_frames = []
    for frame_idx, landmarks in enumerate(pose_landmarks):
        if not landmarks:
            continue
            
        # 获取右肩和右肘的关键点
        shoulder = landmarks[RIGHT_SHOULDER]
        elbow = landmarks[RIGHT_ELBOW]

        # 确保关键点是有效且可见的
        if shoulder.visibility < 0.5 or elbow.visibility < 0.5:
            continue
        
        # 将归一化的坐标转换为像素坐标
        shoulder_px = (shoulder.x * w, shoulder.y * h)
        elbow_px = (elbow.x * w, elbow.y * h)

        # 计算右大臂的向量 (x, y)
        arm_vector_x = elbow_px[0] - shoulder_px[0]
        # 在图像坐标系中，Y轴是向下的，为了与标准数学坐标系对齐，需要反转Y向量
        arm_vector_y = -(elbow_px[1] - shoulder_px[1]) 

        # 使用arctan2计算向量与水平正方向的夹角（弧度）
        angle_rad = np.arctan2(arm_vector_y, arm_vector_x)
        # 将弧度转换为角度
        angle_deg = np.degrees(angle_rad)

        # 判断角度的绝对值是否在15度以内
        if abs(angle_deg) <= 15:
            valid_arm_angle_frames.append(frame_idx)

    # --- 开始评分 ---

    # 规则1: 如果没有任何一帧的右大臂角度符合要求，直接返回60分
    if not valid_arm_angle_frames:
        print("姿势分析：在整个视频中，右大臂均未达到水平准备姿势（±15°）。")
        return 60.0

    print(f"姿势分析：检测到 {len(valid_arm_angle_frames)} 帧的右大臂处于水平准备姿势。")
    
    # 规则2: 在符合条件的帧中，判断篮球位置
    ball_in_quadrant_count = 0
    for frame_idx in valid_arm_angle_frames:
        landmarks = pose_landmarks[frame_idx]
        
        # 获取右耳坐标作为原点
        ear = landmarks[RIGHT_EAR]
        if ear.visibility < 0.5:
            continue  # 如果耳朵不可见，无法判断象限，跳过此帧
        
        ear_px = (ear.x * w, ear.y * h)
        
        # YOLO的帧号从1开始，而我们的索引从0开始，所以需要+1
        frame_num_yolo = frame_idx + 1
        
        # 查找当前帧是否有篮球的检测结果
        ball_pos = ball_by_frame.get(frame_num_yolo)
        if ball_pos is None:
            continue  # 当前帧没有检测到篮球，跳过

        # 判断篮球中心点是否在以右耳为原点的第一象限
        # 条件：ball_x > ear_x 且 ball_y < ear_y (因为Y轴向下)
        if ball_pos['center_x'] > ear_px[0] and ball_pos['center_y'] < ear_px[1]:
            ball_in_quadrant_count += 1
            
    # 计算篮球位置正确的帧在所有有效帧中的占比
    if not valid_arm_angle_frames:
        percentage_in_quadrant = 0.0
    else:
        percentage_in_quadrant = ball_in_quadrant_count / len(valid_arm_angle_frames)
    
    print(f"姿势分析：在准备姿势下，篮球位置正确（位于右耳前方第一象限）的帧数占比为 {percentage_in_quadrant:.2%}")

    # 规则3: 根据占比计算最终得分
    final_score = 60.0 + percentage_in_quadrant * 40.0
    
    return final_score