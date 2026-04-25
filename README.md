# SPA Basketball Shot Analysis

基于计算机视觉的篮球投篮动作分析工具，提供：

- 视频姿态关键点检测
- 篮筐/篮球/人物检测
- 投篮视野、出手角度、准备姿势、上身稳定性、发力顺序评分
- 结果视频可视化
- 雷达图、关节角度图、平滑后关节角度图展示
- Flask 网页前端上传和结果查看

## 项目结构

- `app.py`：Flask Web 服务入口
- `main.py`：统一分析入口，负责组织整条分析流水线
- `templates/index.html`：前端页面
- `keypoint_detector.py`：姿态关键点检测
- `yolo_detector.py`：YOLO 目标检测
- `extract_joint_angle.py`：关节角度提取与平滑
- `shoot_sequence_analysis.py`：发力顺序分析
- `basketball_trajectory.py`：轨迹模拟与绘图
- `release_point_analysis.py`：准备姿势分析
- `upper_body_stability.py`：上身稳定性分析
- `shoot_angle_optimize.py`：出手参数估计
- `view_calculator.py`：视野计算

## 环境准备

建议使用 Python 3.10。

安装依赖：

```bash
pip install -r requirements.txt
```

## 模型文件

仓库默认不包含大模型文件，请自行放置：

- `best0821.pt`
- `yolov8m.pt`（如果你需要切回通用模型）

## 启动方式

运行 Web 服务：

```bash
python app.py
```

默认访问地址：

```text
http://127.0.0.1:5000
```

## 使用说明

1. 打开网页。
2. 上传投篮视频。
3. 等待后端完成分析。
4. 页面会展示：
   - 结果视频
   - 综合总分
   - 分项评分
   - 雷达图
   - 关节角度变化图
   - 平滑后关节角度变化图
   - 发力顺序分析图

## 注意事项

- `dataset/`、`web_runtime/`、模型权重、结果视频和图片默认不提交到 GitHub。
- 当前项目依赖 `mediapipe`、`opencv-python`、`ultralytics`，首次安装可能较慢。
- 如果浏览器内嵌视频不能直接播放，可通过页面中的“新窗口打开视频”链接查看结果视频。

## 适合上传到 GitHub 的必要文件

当前建议上传的主要代码文件包括：

- `app.py`
- `main.py`
- `templates/index.html`
- 所有分析相关 `.py` 模块
- `requirements.txt`
- `README.md`
- `.gitignore`

不建议上传：

- 模型权重文件
- 输入/输出视频
- 运行时结果目录
- 自动生成图片
