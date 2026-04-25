import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from keypoint_detector import detect_keypoints_from_video  # 你的原函数

# ---------------- 1. 提取关键点 ----------------
video_path = r'dataset/input/shoot_with_rim_2.mp4'
all_landmarks, (w, h, fps) = detect_keypoints_from_video(video_path)

# 转换关键点为可用的 np.array 列表 (N帧 x 33点 x 3坐标)
pose_data = []
for lm in all_landmarks:
    if lm is None:
        # 如果该帧没检测到，填充上一帧或零
        if len(pose_data) > 0:
            pose_data.append(pose_data[-1])
        else:
            pose_data.append(np.zeros((33,3)))
        continue
    frame_pts = np.array([[p.x, p.y, p.z] for p in lm])
    pose_data.append(frame_pts)

pose_data = np.array(pose_data)  # shape: (num_frames, 33, 3)

# ---------------- 2. 设置 3D 骨架动画 ----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([-0.5,0.5])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('3D Pose Animation')

# 骨架连接关系（MediaPipe PoseLandmark）
connections = [
    (11,12), (12,24), (11,23), (23,24), (23,25), (24,26),
    (25,27), (26,28), (11,13), (13,15), (12,14), (14,16)
]

# 初始绘制
scatter, = ax.plot([], [], [], 'bo', markersize=5)
lines = [ax.plot([], [], [], 'r')[0] for _ in connections]

# ---------------- 3. 更新每帧 ----------------
def update(frame_idx):
    frame_pts = pose_data[frame_idx]
    xs, ys, zs = frame_pts[:,0], frame_pts[:,1], frame_pts[:,2]

    # 更新关键点
    scatter.set_data(xs, ys)
    scatter.set_3d_properties(zs)

    # 更新骨架线
    for line, (a, b) in zip(lines, connections):
        line.set_data([xs[a], xs[b]], [ys[a], ys[b]])
        line.set_3d_properties([zs[a], zs[b]])

    return scatter, *lines

# ---------------- 4. 动画 ----------------
ani = FuncAnimation(fig, update, frames=len(pose_data), interval=int(1000/fps), blit=True)
plt.show()

# ---------------- 5. 可选：保存为视频 ----------------
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=fps)
# ani.save('pose_animation.mp4', writer=writer)