import pandas as pd
import math

def check_overlap(box1, box2):
    """
    检查两个边框是否重叠。
    边框格式: [x_min, y_min, x_max, y_max]
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    overlap_x = (x_min1 < x_max2) and (x_max1 > x_min2)
    overlap_y = (y_min1 < y_max2) and (y_max1 > y_min2)
    return overlap_x and overlap_y

def find_first_non_overlap_frame(csv_path):
    """
    找出 person 和 ball 第一次不重叠的帧，并返回前一帧和当前帧的信息。

    Args:
        csv_path (str): CSV 文件的路径。

    Returns:
        tuple: (prev_frame_num, prev_person_bbox, prev_ball_bbox,
                curr_frame_num, curr_person_bbox, curr_ball_bbox)
               如果未找到，则返回 (None, None, None, None, None, None)。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        return None, None, None, None, None, None

    frame_numbers = sorted(df['frame'].unique())
    
    prev_person_bbox = None
    prev_ball_bbox = None
    prev_frame_num = None

    for frame_num in frame_numbers:
        current_frame_df = df[df['frame'] == frame_num]
        person_row = current_frame_df[current_frame_df['class'] == 'person']
        ball_row = current_frame_df[current_frame_df['class'] == 'ball']

        if not person_row.empty and not ball_row.empty:
            curr_person_bbox = person_row[['x_min', 'y_min', 'x_max', 'y_max']].iloc[0].tolist()
            curr_ball_bbox = ball_row[['x_min', 'y_min', 'x_max', 'y_max']].iloc[0].tolist()

            if not check_overlap(curr_person_bbox, curr_ball_bbox):
                # 找到 person 和 ball 第一次不重叠的帧
                return prev_frame_num, prev_ball_bbox, prev_person_bbox, frame_num, curr_ball_bbox, curr_person_bbox

            # 更新前一帧的信息
            prev_person_bbox = curr_person_bbox
            prev_ball_bbox = curr_ball_bbox
            prev_frame_num = frame_num

        elif not ball_row.empty:
            # 只有 ball 的情况下更新 ball bbox，但 person 为空
            prev_person_bbox = None
            prev_ball_bbox = ball_row[['x_min', 'y_min', 'x_max', 'y_max']].iloc[0].tolist()
            prev_frame_num = frame_num

    # 如果循环结束仍未找到
    return None, None, None, None, None, None

def calculate_init_state(FPS,file_path):
    # 调用函数并解包四个返回值
    prev_f_num, prev_ball_bbox, prev_person_bbox, curr_f_num, curr_ball_bbox, curr_person_bbox = find_first_non_overlap_frame(file_path)

    dx=(curr_ball_bbox[0]+curr_ball_bbox[2]-prev_ball_bbox[0]-prev_ball_bbox[2])/2
    dy=(curr_ball_bbox[1]+curr_ball_bbox[3]-prev_ball_bbox[1]-prev_ball_bbox[3])/2
    dz=(dx**2+dy**2)**0.5
    dt=1/FPS
    l=abs(((curr_ball_bbox[0]-curr_ball_bbox[2])+(prev_ball_bbox[0]-prev_ball_bbox[2])+(curr_ball_bbox[1]-curr_ball_bbox[3])+(prev_ball_bbox[1]-prev_ball_bbox[3]))/4) # 画面中球平均直径
    c=0.246/l # 篮球直径/画面中球平均直径, 体现现实与画面的比率
    v_init=dz/dt*c # 球出手速度，单位：米/秒
    # theta_init=math.atan(abs(dy/dz))
    theta_init = math.degrees(math.atan2(abs(dy), abs(dz)))

    h_org=abs((curr_person_bbox[1]-curr_person_bbox[3])+(prev_person_bbox[1]-prev_person_bbox[3]))/2 # 预估出手高度
    h_real=h_org*c # 修正出手高度

    return v_init, theta_init, h_real

if __name__ == '__main__':
    # --- 核心：以7号球为参照 ---
    file_path = r"dataset\output\yolo_detections.csv"
    FPS=30
    result=calculate_init_state(FPS,file_path)
    print(result)

# if curr_f_num is not None:
#     print(f"✅ 找到了分离的帧!")
#     print(f"Person和Ball在第 {curr_f_num} 帧第一次不重叠。")
#     print("-" * 30)
#     if prev_f_num is not None:
#         print(f"上一个有ball的帧: ")
#         print(f"  - 帧号: {prev_f_num}")
#         print(f"  - BBox: {prev_ball_bbox}")
#     else:
#         print("这是第一个有ball的帧，没有上一帧。")
#     print("-" * 30)
#     print(f"当前不重叠的帧: ")
#     print(f"  - 帧号: {curr_f_num}")
#     print(f"  - BBox: {curr_ball_bbox}")
# else:
#     print("❌ 在所有给定的帧中，未能找到不重叠的帧。")