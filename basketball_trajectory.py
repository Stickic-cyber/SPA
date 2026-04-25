import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shoot_angle_optimize import calculate_init_state

# 物理常量和篮筐参数，定义为全局变量，避免重复定义
g = 9.8  # 重力加速度 m/s^2
dt = 0.01  # 时间步长 (秒)
ball_mass = 0.62  # 篮球质量 (公斤)
ball_radius = 0.123  # 篮球半径 (米)
air_resistance_coeff = 0.005  # 空气阻力系数

hoop_min_x = 4.123
hoop_max_x = 4.327
hoop_y = 3.05

hoop_org_min_x = 4.0
hoop_org_max_x = 4.45

def simulate_trajectory(initial_height, initial_angle, initial_speed):
    """
    模拟篮球在空中飞行的轨迹。

    Args:
        initial_height (float): 出手高度 (米).
        initial_angle (float): 出手角度 (度).
        initial_speed (float): 初始速度 (米/秒).
    """
    # 将角度从度转换为弧度
    angle_rad = np.deg2rad(initial_angle)

    # 初始速度分量
    vx = initial_speed * np.cos(angle_rad)
    vy = initial_speed * np.sin(angle_rad)

    # 初始位置
    x = 0
    y = initial_height

    # 存储轨迹数据
    trajectory_x = [x]
    trajectory_y = [y]
    
    # 模拟循环
    hoop_pass_x = None 
    
    while y >= 0:
        # 计算当前速度的合力
        v = np.sqrt(vx**2 + vy**2)
        
        # 计算空气阻力
        F_air = -air_resistance_coeff * v**2
        
        # 将空气阻力分解为x和y分量
        F_air_x = F_air * (vx / v)
        F_air_y = F_air * (vy / v)

        # 计算加速度
        ax = F_air_x / ball_mass
        ay = (F_air_y - ball_mass * g) / ball_mass

        # 检查是否接近篮筐高度，并且篮球正在下落
        if not hoop_pass_x and vy < 0 and y <= hoop_y:
            # 找到最接近篮筐高度时的水平位置
            hoop_pass_x = x
            
        # 更新速度和位置
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        # 存储新的位置
        trajectory_x.append(x)
        trajectory_y.append(y)
        
    # 判断是否命中
    is_in = False
    if hoop_pass_x and hoop_min_x <= hoop_pass_x <= hoop_max_x:
        is_in = True
    
    return trajectory_x, trajectory_y, is_in

def run_and_plot_simulations(initial_height, initial_speed, output_path='basketball_trajectory.png', show=False):
    """运行多角度模拟并绘制结果。"""
    
    # 设定角度范围和步长
    angles = np.arange(30, 81, 1)
    
    # 存储结果
    results = []
    successful_angles = []

    plt.figure(figsize=(12, 8))
    
    # 绘制原始篮筐和有效范围
    plt.plot([hoop_org_min_x, hoop_org_max_x], [hoop_y, hoop_y], color='orange', linewidth=5, label='Original Hoop')
    plt.plot([hoop_min_x, hoop_max_x], [hoop_y, hoop_y], color='red', linewidth=5, label='Effective Hoop')
    
    # 用于控制图例只显示一次
    miss_label_added = False
    
    for angle in angles:
        x_coords, y_coords, hit = simulate_trajectory(initial_height, angle, initial_speed)
        
        if hit:
            # 命中：红色，显示具体角度
            plt.plot(x_coords, y_coords, color='red', label=f'{angle}°')
            successful_angles.append(angle)
        else:
            # 未命中：绿色，如果'Miss'图例未添加，则添加
            if not miss_label_added:
                plt.plot(x_coords, y_coords, color='green', label='Miss')
                miss_label_added = True
            else:
                plt.plot(x_coords, y_coords, color='green') # 不添加图例

        # 存储命中结果
        results.append((angle, 'Success' if hit else 'Miss'))
    
    plt.title(f'Basketball Trajectory Simulation (Initial Speed: {initial_speed}m/s)', fontsize=16)
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    # # 绘制结果表格
    # print("\n--- Simulation Results Table ---")
    # print(f"Initial Height: {initial_height} m, Initial Speed: {initial_speed} m/s")
    # print(f"{'Angle (deg)':<15} | {'Result':<10}")
    # print("-" * 28)
    # for angle, result in results:
    #     print(f"{angle:<15} | {result:<10}")
    
    return successful_angles

if __name__ == '__main__':
    file_path = r"dataset\output\yolo_detections.csv"
    FPS=30
    v_init, theta_init, h_real=calculate_init_state(FPS,file_path)
    # 运行整个模拟程序
    initial_height = h_real
    initial_velocity = v_init
    successful_shots = run_and_plot_simulations(initial_height, initial_velocity)

    print(f"\nSuccessful angles: {successful_shots}")
