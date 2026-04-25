import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def compute_eye_view_lines(all_landmarks, frame_width, frame_height, angle_offset_deg=167.5):
    """
    根据全身关键点计算每帧眼耳中点及夹角线终点。
    返回每帧字典：
    {
        'eye_mid': (x, y),
        'ear_mid': (x, y),
        'end_up': (x, y),
        'end_down': (x, y)
    }
    """
    all_points = []

    for lm in all_landmarks:
        if lm is None:
            all_points.append(None)
            continue

        # 眼耳中点（像素坐标）
        left_eye, right_eye = lm[mp_pose.PoseLandmark.LEFT_EYE], lm[mp_pose.PoseLandmark.RIGHT_EYE]
        left_ear, right_ear = lm[mp_pose.PoseLandmark.LEFT_EAR], lm[mp_pose.PoseLandmark.RIGHT_EAR]

        eye_mid = (int((left_eye.x + right_eye.x)/2 * frame_width),
                   int((left_eye.y + right_eye.y)/2 * frame_height))
        ear_mid = (int((left_ear.x + right_ear.x)/2 * frame_width),
                   int((left_ear.y + right_ear.y)/2 * frame_height))

        # 夹角线计算
        vec_x = ear_mid[0] - eye_mid[0]
        vec_y = ear_mid[1] - eye_mid[1]
        base_angle = math.atan2(vec_y, vec_x)
        offset_rad = math.radians(angle_offset_deg)
        line_len = max(frame_width, frame_height) * 1.5

        end_up = (int(eye_mid[0] + line_len * math.cos(base_angle - offset_rad)),
                  int(eye_mid[1] + line_len * math.sin(base_angle - offset_rad)))
        end_down = (int(eye_mid[0] + line_len * math.cos(base_angle + offset_rad)),
                    int(eye_mid[1] + line_len * math.sin(base_angle + offset_rad)))

        all_points.append({
            'eye_mid': eye_mid,
            'ear_mid': ear_mid,
            'end_up': end_up,
            'end_down': end_down
        })

    return all_points
