from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = BASE_DIR / "dataset/input/nba01.mp4"
DEFAULT_YOLO_MODEL = BASE_DIR / "best0821.pt"


def cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def is_inside_angle(O: tuple[int, int], A: tuple[int, int], B: tuple[int, int], C: tuple[int, int]) -> bool:
    ox, oy = O
    ax, ay = A[0] - ox, A[1] - oy
    bx, by = B[0] - ox, B[1] - oy
    cx, cy = C[0] - ox, C[1] - oy
    return cross(ax, ay, cx, cy) * cross(bx, by, cx, cy) < 0


def init_angle_score(theta_init: float, closest_angle: float) -> float:
    difference = abs(closest_angle - theta_init)
    return 60 + 40 * (difference / closest_angle)


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _landmarks_to_3d(pose_landmarks: list[Any]) -> list[np.ndarray | None]:
    pose_landmarks_3d: list[np.ndarray | None] = []
    for frame_landmarks in pose_landmarks:
        if frame_landmarks:
            pose_landmarks_3d.append(np.array([[lm.x, lm.y, lm.z] for lm in frame_landmarks], dtype=np.float64))
        else:
            pose_landmarks_3d.append(None)
    return pose_landmarks_3d


def draw_radar_chart(
    shoot_view_score: float,
    angle_score: float,
    shooting_form_score: float,
    upper_stability_score: float,
    power_sequence_score: float,
    output_path: str | Path,
) -> float:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["投篮视野", "角度与力度", "准备姿势", "上身稳定性", "发力顺序", "综合总分"]
    scores = [
        _to_float(shoot_view_score),
        _to_float(angle_score),
        _to_float(shooting_form_score),
        _to_float(upper_stability_score),
        _to_float(power_sequence_score),
    ]
    final_score = float(np.mean(scores))
    all_scores = scores + [final_score]

    values = np.concatenate((all_scores, [all_scores[0]]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, color="#0f766e", linewidth=2, linestyle="solid", label="评分")
    ax.fill(angles, values, color="#5eead4", alpha=0.35)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10, color="gray")

    for angle, score in zip(angles[:-1], all_scores):
        ax.text(angle, min(score + 4, 100), f"{score:.1f}", ha="center", va="center", fontsize=10, fontweight="bold")

    plt.title("投篮技术综合评分雷达图", size=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return final_score


def _calculate_rim_in_view(view_line_points: list[Any], yolo_df: pd.DataFrame) -> list[bool]:
    df_rim = yolo_df[yolo_df["class"] == "rim"]
    rim_by_frame = {frame: group.to_dict(orient="records") for frame, group in df_rim.groupby("frame")}

    rim_in_view_list: list[bool] = []
    for i, points in enumerate(view_line_points):
        in_view = False
        if points:
            O, A, B = points["eye_mid"], points["end_up"], points["end_down"]
            for rim_bbox in rim_by_frame.get(i + 1, []):
                top_left = (_to_int(rim_bbox["x_min"]), _to_int(rim_bbox["y_min"]))
                top_right = (_to_int(rim_bbox["x_max"]), _to_int(rim_bbox["y_min"]))
                if is_inside_angle(O, A, B, top_left) and is_inside_angle(O, A, B, top_right):
                    in_view = True
                    break
        rim_in_view_list.append(in_view)
    return rim_in_view_list


def _render_final_video(
    input_video: str | Path,
    output_video: str | Path,
    fps: int,
    width: int,
    height: int,
    view_line_points: list[Any],
    rim_in_view_list: list[bool],
    yolo_df: pd.DataFrame,
) -> None:
    yolo_by_frame = {frame: group.to_dict(orient="records") for frame, group in yolo_df.groupby("frame")}

    cap = cv2.VideoCapture(str(input_video))
    out = None
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            out = writer
            print(f"结果视频编码器: {codec}")
            break
        writer.release()

    if out is None:
        cap.release()
        raise RuntimeError("无法创建结果视频文件，请检查 OpenCV 视频编码器支持。")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(view_line_points) and view_line_points[frame_idx]:
            points = view_line_points[frame_idx]
            eye_mid = points["eye_mid"]
            end_up = points["end_up"]
            end_down = points["end_down"]
            cv2.line(frame, eye_mid, end_up, (0, 255, 255), 2)
            cv2.line(frame, eye_mid, end_down, (0, 255, 255), 2)
            cv2.circle(frame, eye_mid, 5, (255, 0, 255), -1)

        if frame_idx < len(rim_in_view_list):
            status = rim_in_view_list[frame_idx]
            color = (0, 255, 0) if status else (0, 0, 255)
            cv2.putText(
                frame,
                f"Rim in view: {status}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

        for det in yolo_by_frame.get(frame_idx + 1, []):
            x1, y1, x2, y2 = (_to_int(det["x_min"]), _to_int(det["y_min"]), _to_int(det["x_max"]), _to_int(det["y_max"]))
            label = f'{det["class"]} {_to_float(det["confidence"]):.2f}'
            box_color = (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def analyze_video(
    video_file: str | Path,
    output_dir: str | Path = "dataset/output",
    yolo_model: str | Path = DEFAULT_YOLO_MODEL,
) -> dict[str, Any]:
    from basketball_trajectory import run_and_plot_simulations
    from extract_joint_angle import extract_joint_angles, process_angles, save_joint_angle_plot
    from keypoint_detector import detect_keypoints_from_video
    from release_point_analysis import analyze_shooting_form
    from shoot_angle_optimize import calculate_init_state
    from shoot_sequence_analysis import evaluate_power_sequence
    from upper_body_stability import calculate_upper_stability
    from view_calculator import compute_eye_view_lines
    from yolo_detector import detect_on_video_yolov8

    output_dir = _ensure_output_dir(output_dir)
    video_file = Path(video_file)
    yolo_model = Path(yolo_model)
    if not video_file.is_absolute():
        video_file = BASE_DIR / video_file
    if not yolo_model.is_absolute():
        yolo_model = BASE_DIR / yolo_model

    output_csv = output_dir / "yolo_detections.csv"
    yolo_temp_video = output_dir / "temp_yolo_only_video.mp4"
    final_output_video = output_dir / "final_combined_video.mp4"
    radar_chart_path = output_dir / "scoring_radar_chart.png"
    trajectory_chart_path = output_dir / "basketball_trajectory.png"
    raw_joint_angles_chart = output_dir / "joint_angles_raw.png"
    smooth_joint_angles_chart = output_dir / "joint_angles_smoothed.png"
    power_sequence_chart = output_dir / "joint_angles_and_acceleration_plot.png"

    print("视频处理步骤 1/6: 正在检测姿态关键点...")
    pose_landmarks, video_info = detect_keypoints_from_video(str(video_file))
    width, height, fps = video_info
    pose_landmarks_3d = _landmarks_to_3d(pose_landmarks)

    print("视频处理步骤 2/6: 正在计算视角线...")
    view_line_points = compute_eye_view_lines(pose_landmarks, width, height)

    print("视频处理步骤 3/6: 正在执行 YOLO 检测...")
    detect_on_video_yolov8(
        str(video_file),
        model_name=str(yolo_model),
        output_csv=str(output_csv),
        output_video=str(yolo_temp_video),
    )
    yolo_df = pd.read_csv(output_csv)

    print("视频处理步骤 4/6: 正在判断篮筐是否进入视野...")
    rim_in_view_list = _calculate_rim_in_view(view_line_points, yolo_df)

    print("视频处理步骤 5/6: 正在渲染结果视频...")
    _render_final_video(
        video_file,
        final_output_video,
        fps,
        width,
        height,
        view_line_points,
        rim_in_view_list,
        yolo_df,
    )

    print("视频处理步骤 6/6: 正在生成关节角度图...")
    angles_dict = extract_joint_angles(pose_landmarks, video_info)
    filtered_angles_dict = process_angles(angles_dict)
    save_joint_angle_plot(angles_dict, fps, raw_joint_angles_chart, "关节角度变化图")
    save_joint_angle_plot(filtered_angles_dict, fps, smooth_joint_angles_chart, "关节角度变化图（平滑处理后）")

    true_count = sum(rim_in_view_list)
    total_count = len(rim_in_view_list)
    true_percentage = (true_count / total_count) * 100 if total_count else 0.0
    shoot_view_score = 60 + true_percentage * 0.4 if total_count else 60.0

    print("评分分析 1/5: 正在计算出手角度与力度...")
    angle_score = 60.0
    recommended_angle_range: list[float] = []
    try:
        v_init, theta_init, h_real = calculate_init_state(fps, str(output_csv))
    except Exception:
        v_init, theta_init, h_real = None, None, None
    if None not in (v_init, theta_init, h_real):
        successful_shots = run_and_plot_simulations(
            h_real,
            v_init + 1,
            output_path=str(trajectory_chart_path),
            show=False,
        )
        recommended_angle_range = [float(successful_shots[0]), float(successful_shots[-1])] if successful_shots else []
        if successful_shots:
            if theta_init < successful_shots[0]:
                angle_score = init_angle_score(theta_init, successful_shots[0])
            elif theta_init > successful_shots[-1]:
                angle_score = init_angle_score(theta_init, successful_shots[-1])
            else:
                angle_score = 95.0

    print("评分分析 2/5: 正在计算准备姿势得分...")
    shooting_form_score = _to_float(analyze_shooting_form(pose_landmarks, yolo_df, video_info), 60.0)

    print("评分分析 3/5: 正在计算上身稳定性...")
    Q1, median_x, Q3, in_iqr_percentage = calculate_upper_stability(pose_landmarks)
    stability_score = 60 + 0.4 * _to_float(in_iqr_percentage, 0.0)

    print("评分分析 4/5: 正在计算发力顺序...")
    power_sequence_score = _to_float(
        evaluate_power_sequence(pose_landmarks_3d, frame_rate=fps, plot_output_path=str(power_sequence_chart)),
        60.0,
    )

    print("评分分析 5/5: 正在生成雷达图...")
    final_score = draw_radar_chart(
        shoot_view_score,
        angle_score,
        shooting_form_score,
        stability_score,
        power_sequence_score,
        radar_chart_path,
    )

    return {
        "video_info": {"width": width, "height": height, "fps": fps},
        "scores": {
            "shoot_view_score": round(shoot_view_score, 2),
            "angle_score": round(angle_score, 2),
            "shooting_form_score": round(shooting_form_score, 2),
            "stability_score": round(stability_score, 2),
            "power_sequence_score": round(power_sequence_score, 2),
            "final_score": round(final_score, 2),
        },
        "metrics": {
            "rim_in_view_percentage": round(true_percentage, 2),
            "upper_body_q1": round(_to_float(Q1), 4),
            "upper_body_median": round(_to_float(median_x), 4),
            "upper_body_q3": round(_to_float(Q3), 4),
            "upper_body_iqr_percentage": round(_to_float(in_iqr_percentage), 2),
            "initial_speed": round(_to_float(v_init), 4),
            "initial_angle": round(_to_float(theta_init), 4),
            "release_height": round(_to_float(h_real), 4),
            "recommended_angle_range": recommended_angle_range,
        },
        "assets": {
            "video": str(final_output_video),
            "radar_chart": str(radar_chart_path),
            "trajectory_chart": str(trajectory_chart_path),
            "raw_joint_angles_chart": str(raw_joint_angles_chart),
            "smoothed_joint_angles_chart": str(smooth_joint_angles_chart),
            "power_sequence_chart": str(power_sequence_chart),
            "yolo_csv": str(output_csv),
        },
    }


def main() -> None:
    result = analyze_video(DEFAULT_VIDEO, output_dir="dataset/output", yolo_model=DEFAULT_YOLO_MODEL)
    print("分析完成。")
    print(result["scores"])


if __name__ == "__main__":
    main()
