import cv2
import pandas as pd
from ultralytics import YOLO
import time
import numpy as np

def detect_on_video_yolov8(video_path, model_name='yolov8m.pt', output_csv='video_results_yolov8.csv', output_video='output_video_yolov8.mp4'):
    """
    使用 YOLOv8 模型识别视频中每一帧的目标，
    将结果保存到 CSV，并将带有边界框的视频保存下来。
    
    Args:
        video_path (str): 待处理的视频文件路径。
        model_name (str): YOLO 模型名称或路径 (例如 'yolov8n.pt', 'yolov8s.pt')。
        output_csv (str): 结果保存的 CSV 文件路径。
        output_video (str): 输出视频的保存路径。
    """
    # 初始化 YOLOv8 模型
    print(f"正在加载模型 {model_name}...")
    model = YOLO(model_name)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件: {video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建 VideoWriter 对象以保存输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # COCO 数据集中 'person' (0) 和 'sports ball' (32) 的类别 ID 
    # 微调后：0: ball  1: made 2: person  3: rim  4: shoot
    target_class_ids = [0, 1, 2, 3, 4]
    
    all_results_data = []
    frame_number = 0
    start_time = time.time()

    print("开始逐帧处理视频...")
    
    # 循环处理视频的每一帧
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("YOLO识别完毕。")
            break
        
        frame_number += 1
        
        # 在当前帧上进行目标检测
        results = model(frame, conf=0.5, classes=target_class_ids, verbose=False)
        
        annotated_frame = frame.copy()
        
        # 创建一个字典来存储每个类别置信度最高的结果
        best_results_per_class = {}
        
        # 遍历所有检测结果
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 如果当前类别还没有结果，或者当前结果置信度更高，则更新
                if cls_id not in best_results_per_class or conf > best_results_per_class[cls_id]['confidence']:
                    best_results_per_class[cls_id] = {
                        'box': box,
                        'confidence': conf
                    }

        # 遍历每个类别中置信度最高的结果，并绘制和保存
        for cls_id, data in best_results_per_class.items():
            box = data['box']
            conf = data['confidence']
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = model.names[cls_id]

            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            label = f'{class_name}: {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x1 + w, y1 - h - 5), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 将结果添加到列表中
            all_results_data.append({
                'frame': frame_number,
                'class': class_name,
                'confidence': round(conf, 2),
                'x_min': x1,
                'y_min': y1,
                'x_max': x2,
                'y_max': y2
            })
            
        # 将绘制好边界框的帧写入输出视频
        out.write(annotated_frame)
        
        # 打印进度
        if frame_number % 30 == 0:
            print(f"已处理 {frame_number} 帧...")

    end_time = time.time()

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 保存检测结果到 CSV
    if all_results_data:
        df = pd.DataFrame(all_results_data)
        df.to_csv(output_csv, index=False)
        print(f"检测结果已保存到 {output_csv}")
    else:
        print("视频中未检测到目标。")

    print(f"处理后的视频已保存到 {output_video}")
    print(f"总耗时: {end_time - start_time:.2f} 秒，共处理 {frame_number} 帧。")


if __name__ == '__main__':
    # --- 指定要使用的视频文件 ---
    video_file = r'D:\Project\SPA2_0\video_scrawler\download\A. Black Free Throw 1 of 2 (7 PTS) 3 03_49 Fri Oct 24 2025.mp4'
    
    # --- 执行函数，使用 YOLOv8 模型 ---
    detect_on_video_yolov8(video_file, model_name=r'best0821.pt', output_csv=r'dataset\output\video_results_yolov8.csv', output_video=r'dataset\output\output.mp4')
