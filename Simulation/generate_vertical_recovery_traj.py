"""
基于 trajectory_01.csv 生成带姿态摆动的回收轨迹。

在原始轨迹的位置 / 速度 / 加速度不变的前提下，
给 roll、pitch 叠加小幅垂直偏转摆动，给 yaw 叠加绕箭体中心轴的缓慢旋转，
模拟回收过程中的真实箭体姿态扰动。近地面时摆动幅度自动衰减至零。

输出 CSV: time,x,y,z,roll,pitch,yaw,vx,vy,vz,ax,ay,az  (角度单位: rad)
"""

import argparse
import csv
import math
import os


def generate(src_path: str, out_path: str) -> None:
    with open(src_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    z_max = float(rows[0]['z'])

    result = []
    for row in rows:
        t = float(row['time'])
        z = float(row['z'])
        roll0  = float(row['roll'])
        pitch0 = float(row['pitch'])
        yaw0   = float(row['yaw'])

        # 摆动幅度包络: 高空满幅，近地面 (z < 60 m) 线性衰减至零
        envelope = min(1.0, z / 60.0) * math.sqrt(min(z / z_max, 1.0))

        # 垂直方向小幅偏转 (roll / pitch 摆动)
        d_roll  = math.radians(1.8) * envelope * math.sin(0.95 * t + 0.40)
        d_pitch = math.radians(1.5) * envelope * math.sin(1.20 * t + 1.10)

        # 绕中心轴缓慢旋转 + 小幅振荡
        yaw_drift = math.radians(3.0) * envelope * math.sin(0.35 * t + 0.20)
        d_yaw = yaw_drift + math.radians(1.0) * envelope * math.sin(0.80 * t + 2.50)

        result.append({
            'time': row['time'],
            'x':    row['x'],
            'y':    row['y'],
            'z':    row['z'],
            'roll':  f"{roll0 + d_roll:.6f}",
            'pitch': f"{pitch0 + d_pitch:.6f}",
            'yaw':   f"{yaw0 + d_yaw:.6f}",
            'vx':   row['vx'],
            'vy':   row['vy'],
            'vz':   row['vz'],
            'ax':   row['ax'],
            'ay':   row['ay'],
            'az':   row['az'],
        })

    cols = ['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
            'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(result)

    print(f"[OK] 已生成轨迹: {out_path}  ({len(result)} 帧)")


if __name__ == '__main__':
    here = os.path.dirname(__file__) or '.'
    parser = argparse.ArgumentParser(
        description='在已有轨迹上叠加姿态摆动，生成 traj.csv')
    parser.add_argument(
        '--src', type=str,
        default=os.path.join(here, 'trajectory_01.csv'),
        help='源轨迹 CSV (默认: trajectory_01.csv)')
    parser.add_argument(
        '--out', type=str,
        default=os.path.join(here, 'traj.csv'),
        help='输出 CSV (默认: traj.csv)')
    args = parser.parse_args()

    generate(os.path.abspath(args.src), os.path.abspath(args.out))
