import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

# --- 以下是您提供的先进追踪算法的完整实现 ---
# --- (作为内部模块，无需修改) ---

def moving_least_square_numpy(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """1-D Moving Least Squares"""
    p = np.stack([np.ones_like(x), x], axis=-2)
    M = p @ (w[..., :, None] * p.swapaxes(-2, -1))
    # 使用伪逆来增加数值稳定性，防止M矩阵奇异
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)
    a = M_inv @ (p @ (w * y)[..., :, None])
    a = a.squeeze(-1)
    return a

def intrinsic_from_fov(fov: float, width: int, height: int) -> np.ndarray:
    """从FOV计算相机内参矩阵"""
    px, py = width / 2, height / 2
    fx = px / np.tan(fov / 2)
    fy = py / np.tan(fov / 2)
    return np.array([
        [fx, 0., px],
        [0., fy, py],
        [0., 0., 1.],
    ], dtype=np.float32)

def mls_smooth_numpy(input_t: List[float], input_y: List[np.ndarray], query_t: float, smooth_range: float):
    """使用MLS平滑时间序列数据"""
    if not input_y:
        return None
    if len(input_y) == 1:
        return input_y[0]
    
    input_t = np.array(input_t) - query_t
    input_y = np.stack(input_y, axis=-1)
    broadcaster = (None,) * (len(input_y.shape) - 1)
    
    # 计算权重，确保至少有一个非零权重以避免除以零
    w = np.maximum(smooth_range - np.abs(input_t), 0)
    if np.sum(w) < 1e-6: # 如果所有权重都接近零，返回最近的那个点
        return input_y[..., np.argmin(np.abs(input_t))]

    coef = moving_least_square_numpy(input_t[broadcaster], input_y, w[broadcaster])
    return coef[..., 0]

MEDIAPIPE_POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]

WEIGHTS = {
    'left_ear': 0.04, 'right_ear': 0.04, 'left_shoulder': 0.18, 'right_shoulder': 0.18,
    'left_elbow': 0.02, 'right_elbow': 0.02, 'left_wrist': 0.01, 'right_wrist': 0.01,
    'left_hip': 0.2, 'right_hip': 0.2, 'left_knee': 0.03, 'right_knee': 0.03,
    'left_ankle': 0.02, 'right_ankle': 0.02,
}

class BodyKeypointTrack:
    """一个集成了PnP和MLS平滑的3D关键点追踪器类"""
    def __init__(self, im_width: int, im_height: int, fov: float, frame_rate: float, *, model_complexity=1, smooth_range: float = 0.3, smooth_range_barycenter: float = 1.0):
        self.K = intrinsic_from_fov(fov, im_width, im_height)
        self.im_width, self.im_height = im_width, im_height
        self.frame_delta = 1. / frame_rate

        self.mp_pose_model = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_rvec, self.pose_tvec = None, None
        self.barycenter_weight = np.array([WEIGHTS.get(kp, 0.) for kp in MEDIAPIPE_POSE_KEYPOINTS])

        self.smooth_range = smooth_range
        self.smooth_range_barycenter = smooth_range_barycenter
        self.barycenter_history: List[Tuple[np.ndarray, float]] = []
        self.pose_history: List[Tuple[np.ndarray, float]] = []

    def _get_camera_space_landmarks(self, image_landmarks, world_landmarks, visible, rvec, tvec):
        _, rvec, tvec = cv2.solvePnP(world_landmarks[visible], image_landmarks[visible], self.K, np.zeros(5), rvec=rvec, tvec=tvec, useExtrinsicGuess=rvec is not None, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rvec)
        kpts3d_cam = world_landmarks @ rmat.T + tvec.T
        kpts3d_cam_z = kpts3d_cam[:, 2].reshape(-1, 1)
        kpts3d_cam[:, :2] = (np.concatenate([image_landmarks, np.ones((image_landmarks.shape[0], 1))], axis=1) @ np.linalg.inv(self.K).T * kpts3d_cam_z)[:, :2]
        return kpts3d_cam, rvec, tvec

    def track(self, image: np.ndarray, t: float):
        results = self.mp_pose_model.process(image)
        if results.pose_landmarks is None or results.pose_world_landmarks is None:
            return

        image_landmarks = np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in results.pose_landmarks.landmark])
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
        visible = np.array([lm.visibility > 0.2 for lm in results.pose_landmarks.landmark])

        if visible.sum() < 6:
            return
        
        kpts3d, rvec, tvec = self._get_camera_space_landmarks(image_landmarks, world_landmarks, visible, self.pose_rvec, self.pose_tvec)
        if tvec[2] < 0: # 过滤掉相机后方的点
            return

        self.pose_rvec, self.pose_tvec = rvec, tvec
        barycenter = np.average(kpts3d, axis=0, weights=self.barycenter_weight)
        self.barycenter_history.append((barycenter, t))
        self.pose_history.append((kpts3d - barycenter, t))

    def get_smoothed_3d_keypoints(self, query_t: float):
        barycenter_list = [barycenter for barycenter, t in self.barycenter_history if abs(t - query_t) < self.smooth_range_barycenter]
        barycenter_t = [t for _, t in self.barycenter_history if abs(t - query_t) < self.smooth_range_barycenter]
        barycenter = mls_smooth_numpy(barycenter_t, barycenter_list, query_t, self.smooth_range_barycenter)
        if barycenter is None:
            return np.zeros((len(MEDIAPIPE_POSE_KEYPOINTS), 3)), np.zeros(len(MEDIAPIPE_POSE_KEYPOINTS), dtype=bool)

        pose_kpts3d_list = [kpts3d for kpts3d, t in self.pose_history if abs(t - query_t) < self.smooth_range]
        pose_t = [t for _, t in self.pose_history if abs(t - query_t) < self.smooth_range]
        
        # 检查当前帧附近是否有有效数据
        has_recent_data = any(abs(t - query_t) < self.frame_delta * 0.6 for t in pose_t)
        if not has_recent_data:
            return np.zeros((len(MEDIAPIPE_POSE_KEYPOINTS), 3)), np.zeros(len(MEDIAPIPE_POSE_KEYPOINTS), dtype=bool)

        pose_kpts3d = mls_smooth_numpy(pose_t, pose_kpts3d_list, query_t, self.smooth_range)
        if pose_kpts3d is None:
            return np.zeros((len(MEDIAPIPE_POSE_KEYPOINTS), 3)), np.zeros(len(MEDIAPIPE_POSE_KEYPOINTS), dtype=bool)

        all_kpts3d = pose_kpts3d + barycenter.reshape(1, 3)
        all_valid = np.full(len(MEDIAPIPE_POSE_KEYPOINTS), True)
        return all_kpts3d, all_valid

# --- 以上部分作为内部实现 ---

# --- 下面是对外暴露的标准接口函数 ---
mp_pose = mp.solutions.pose

def detect_keypoints_from_video(video_path):
    """
    检测全身关键点，每帧返回 landmark 列表（x, y, z, visibility）。
    内部使用先进的3D追踪与平滑算法，但输出格式保持不变。
    """
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # 防止视频文件fps信息错误
        fps = 30 

    # 1. 初始化原始的MediaPipe模型，用于获取z和visibility的“模板”
    pose_template_provider = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.5
    )

    # 2. 初始化高级追踪器
    # 默认FOV设为60度 (pi/4)，与之前请求一致
    body_keypoint_tracker = BodyKeypointTrack(
        w, h, fov=np.pi / 4, frame_rate=fps, model_complexity=1
    )

    all_landmarks = []
    frame_t = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. 运行原始模型获取基准结果
        original_results = pose_template_provider.process(rgb)

        if original_results.pose_landmarks:
            # 4. 将当前帧喂给高级追踪器
            body_keypoint_tracker.track(rgb, frame_t)
            
            # 5. 从追踪器获取平滑后的3D坐标
            kpts3d, visib = body_keypoint_tracker.get_smoothed_3d_keypoints(frame_t)
            
            if visib.any(): # 如果追踪器返回了有效结果
                # 6. 将3D坐标投影回2D像素坐标
                K = body_keypoint_tracker.K
                kpts3d_homo = kpts3d @ K.T
                
                # 防止除以0（点在相机后方或平面上）
                kpts3d_homo_z = kpts3d_homo[:, 2:]
                kpts3d_homo_z[np.abs(kpts3d_homo_z) < 1e-6] = 1e-6
                kpts2d = kpts3d_homo[:, :2] / kpts3d_homo_z

                # 7. 回填到原始landmark列表中，保证格式完全兼容
                output_landmarks = original_results.pose_landmarks.landmark
                for i in range(len(output_landmarks)):
                    if visib[i]: # 只更新追踪器认为有效的点
                        output_landmarks[i].x = kpts2d[i, 0] / w
                        output_landmarks[i].y = kpts2d[i, 1] / h
                        # .z 和 .visibility 保持原始值不变
                
                all_landmarks.append(output_landmarks)
            else:
                # 如果追踪器没有有效输出，则回退到使用原始检测结果
                all_landmarks.append(original_results.pose_landmarks.landmark)
        else:
            # 如果原始模型未检测到，则为None
            all_landmarks.append(None)
        
        # 更新时间戳
        frame_t += 1.0 / fps

    cap.release()
    pose_template_provider.close()
    body_keypoint_tracker.mp_pose_model.close()
    return all_landmarks, (w, h, int(fps))