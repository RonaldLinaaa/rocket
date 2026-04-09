"""
Module 1b — Falcon 9 一级着陆段轨迹仿真器 (Python)
可复用火箭回收着陆段物理仿真轨迹生成 — 理想情况

物理模型:
  ┌──────────────────────────────────────────────────────────────┐
  │ 阶段 1: 气动减速段 (3000 m → 点火高度)                      │
  │   • 仅重力 + 气动阻力，无推力，速度单调减小                  │
  │   • 栅格翼提供有限水平修正力 (PD 控制)                       │
  ├──────────────────────────────────────────────────────────────┤
  │ 阶段 2: 动力制动段 (点火高度 → 0 m)                         │
  │   • 点火后推力线性爬升 T_THRUST_RAMP 秒，避免加速度突变      │
  │   • 垂直: 线性参考速度剖面 v_ref(z)，从点火速度平滑过渡到触地 │
  │   • 水平: ZEM/ZEV 制导收敛至着陆点中心                       │
  │   • 燃料质量实时递减，落地速度 ≈ VZ_LAND (可设为 0)           │
  ├──────────────────────────────────────────────────────────────┤
  │ 终端条件                                                     │
  │   • 位置: (0, 0, 0) — 着陆标志物中心                         │
  │   • 速度: vz ≈ −2 m/s,  vx ≈ 0,  vy ≈ 0                    │
  └──────────────────────────────────────────────────────────────┘

Falcon 9 Block 5 一级参考参数:
  干质量 ≈ 25 800 kg  |  着陆剩余燃料 ≈ 7 000 kg
  Merlin 1D 海平面推力 854 kN  |  海平面比冲 282 s
  箭体直径 3.66 m

坐标系: X = East, Y = North, Z = Up,  原点 = 着陆点标志物中心
输出:   trajectory.csv  (50 Hz, 13 列)
        time, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az
"""

import math
import csv

# ═══════════════════════════════════════════════════════════════════
#  物理常量
# ═══════════════════════════════════════════════════════════════════
G       = 9.81          # 重力加速度 (m/s²)
DT      = 0.02          # 积分步长 (s), 对应 50 Hz
RHO_SL  = 1.225         # 海平面标准空气密度 (kg/m³)
H_SCALE = 8500.0        # 大气标高 (m)

# ═══════════════════════════════════════════════════════════════════
#  Falcon 9 一级箭体参数
# ═══════════════════════════════════════════════════════════════════
M_DRY       = 25_800.0       # 干质量 (kg)
M_FUEL      = 7_000.0        # 着陆段可用燃料 (kg)
THRUST_MAX  = 854_000.0      # Merlin 1D 海平面最大推力 (N)
THROTTLE_MIN = 0.40          # 最低节流比 (物理限制, 理想情况下放宽至 0)
ISP_SL      = 282.0          # 海平面比冲 (s)
DIAMETER    = 3.66           # 箭体直径 (m)
CD_BODY     = 1.3            # 底部朝前阻力系数 (含栅格翼展开)
A_REF       = math.pi * (DIAMETER / 2) ** 2   # 参考截面积 (m²)

# ═══════════════════════════════════════════════════════════════════
#  初始条件
# ═══════════════════════════════════════════════════════════════════
Z0      = 3000.0                       # 初始高度 (m)
V0      = 200.0                # 1000 km/h → 277.78 m/s
GAMMA0  = math.radians(3.0)           # 飞行路径偏离竖直 (rad), 近垂直

VZ0     = -V0 * math.cos(GAMMA0)      # 垂直分量 ≈ −277.4 m/s
VH0     =  V0 * math.sin(GAMMA0)      # 水平分量 ≈  14.5 m/s

X0, Y0  = 50.0, -25.0                 # 初始水平偏移 (m)
_D0     = math.sqrt(X0 ** 2 + Y0 ** 2)
VX0     = -VH0 * X0 / _D0 if _D0 > 1 else 0.0
VY0     = -VH0 * Y0 / _D0 if _D0 > 1 else 0.0

VZ_LAND = -0.0                        # 触地目标垂直速度 (m/s)

# 点火高度 (m): 火箭下降至此高度时发动机点火
# 设为 None 时使用能量约束自动判据; 设为数值 (如 2500) 则固定在该高度点火
Z_IGNITION_ALT = 2500.0

# 点火后推力爬升时间 (s): 该时间内推力从 0 线性增至制导指令值，使加速度连续、速度平滑
T_THRUST_RAMP = 0.5


# ═══════════════════════════════════════════════════════════════════
#  气动与大气模型
# ═══════════════════════════════════════════════════════════════════

def atm_density(z):
    """指数大气密度: ρ(z) = ρ_SL · exp(−z / H)"""
    return RHO_SL * math.exp(-max(z, 0.0) / H_SCALE)


def aero_drag_accel(vx, vy, vz, z, mass):
    """
    气动阻力加速度 (各轴分量).
    F_drag = ½ ρ V² C_D A_ref, 方向与速度相反.
    """
    v = math.sqrt(vx * vx + vy * vy + vz * vz)
    if v < 0.01:
        return 0.0, 0.0, 0.0
    rho = atm_density(z)
    a_drag = 0.5 * rho * v * v * CD_BODY * A_REF / mass
    inv_v = 1.0 / v
    return -a_drag * vx * inv_v, -a_drag * vy * inv_v, -a_drag * vz * inv_v


# ═══════════════════════════════════════════════════════════════════
#  制导: 点火判据 + 恒减速剖面 + ZEM/ZEV 水平修正
# ═══════════════════════════════════════════════════════════════════

def should_ignite(z, vz, z_alt, vx, vy, mass):
    """
    基于能量约束的点火判据.
    计入当前气动阻力贡献, 采用 15% 安全裕度.
    """
    v_down = abs(vz)
    v_total = math.sqrt(vx * vx + vy * vy + vz * vz)

    # 当前气动减速贡献
    rho = atm_density(z_alt)
    drag_decel = 0.5 * rho * v_total * v_total * CD_BODY * A_REF / mass
    a_avail = THRUST_MAX / mass - G + drag_decel * 0.5   # 保守取一半阻力贡献

    if a_avail <= 0:
        return True
    h_stop = (v_down ** 2 - VZ_LAND ** 2) / (2.0 * a_avail)
    return z_alt <= h_stop * 1.15


def powered_guidance(x, y, z, vx, vy, vz, z_ignition=None, vz_ignition=None):
    """
    着陆制导律 — 全程减速、速度平滑:
      垂直 — 线性参考速度剖面: v_ref(z) 从点火时 vz 线性过渡到触地 VZ_LAND
      水平 — ZEM/ZEV 多项式制导, 带低空增益衰减

    z_ignition, vz_ignition: 点火时刻高度与垂直速度，用于构造平滑参考剖面 (m, m/s)
    返回推力加速度指令 (ax, ay, az).
    """
    v_down = abs(vz)

    # ─ 垂直: 平滑参考剖面 (前段省推力、后段强制动) + 比例跟踪 ─
    if z > 1.0 and z_ignition is not None and vz_ignition is not None and z_ignition > 1.0:
        # 使用 frac^0.7: 前期减速较缓省燃料，后期减速加快确保落地接近 0
        frac = min(1.0, max(0.0, z / z_ignition))
        frac_power = frac ** 0.7
        vz_ref = VZ_LAND + (vz_ignition - VZ_LAND) * frac_power
        # d(vz_ref)/dz = (vz_ignition - VZ_LAND) * 0.7 * frac^(-0.3) / z_ignition
        dv_ref_dz = (vz_ignition - VZ_LAND) * 0.7 * (frac ** (-0.3)) / z_ignition if frac > 1e-6 else 0.0
        a_net_ref = dv_ref_dz * vz
        Kp = 2.0 if z < 50.0 else (1.5 if z < 120.0 else 1.0)  # 低空加强跟踪
        az_thrust = a_net_ref + G + Kp * (vz_ref - vz)
        # 落后剖面时施加最小制动，避免落地过快
        a_brake_min = (v_down ** 2 - VZ_LAND ** 2) / (2.0 * z)
        if a_brake_min > 0 and vz < vz_ref - 1.5:
            a_net_need = a_brake_min * 1.02
            az_thrust = max(az_thrust, G + a_net_need)
        if vz < 0 and az_thrust < G + 0.5:
            az_thrust = G + 0.5
        t_go = 2.0 * z / max(v_down + abs(VZ_LAND), 0.1) if v_down + abs(VZ_LAND) > 0.1 else 10.0
    elif z > 1.0:
        # 无点火状态时 fallback: 恒减速剖面
        a_brake_net = (v_down ** 2 - VZ_LAND ** 2) / (2.0 * z)
        vz_profile = -math.sqrt(max(VZ_LAND ** 2 + 2.0 * a_brake_net * z, VZ_LAND ** 2))
        az_thrust = a_brake_net + G + 0.8 * (vz_profile - vz)
        t_go = (v_down - abs(VZ_LAND)) / a_brake_net if a_brake_net > 0.1 else 2.0 * z / max(v_down, 0.1)
    else:
        # 极低空: 直接收敛到触地速度
        az_thrust = G + 3.0 * (VZ_LAND - vz)
        t_go = 0.5

    t_go = max(0.5, t_go)
    t_go2 = t_go * t_go

    # ─ 水平 ZEM/ZEV 制导, 低空线性衰减 ─
    gain_h = min(1.0, z / 15.0) if z < 15.0 else 1.0
    ax_thrust = gain_h * (-(6.0 / t_go2) * (x + vx * t_go) - (2.0 / t_go) * vx)
    ay_thrust = gain_h * (-(6.0 / t_go2) * (y + vy * t_go) - (2.0 / t_go) * vy)
    if z < 30.0:
        ax_thrust -= 2.0 * vx
        ay_thrust -= 2.0 * vy

    return ax_thrust, ay_thrust, az_thrust


def apply_thrust_limits(ax, ay, az, mass):
    """
    推力限幅: 优先保证垂直制动。超限时先缩水平分量使总推力 = T_max，
    若垂直单独已超限则整体同比例缩小。
    """
    a_mag = math.sqrt(ax * ax + ay * ay + az * az)
    if a_mag < 1e-9:
        return 0.0, 0.0, 0.0, 0.0
    F_cmd = a_mag * mass
    if F_cmd <= THRUST_MAX:
        return ax, ay, az, F_cmd

    # 超限: 保持垂直 az 不变，缩小 ax,ay 使 |F| = THRUST_MAX
    a_lat2 = ax * ax + ay * ay
    a_max = THRUST_MAX / mass
    if a_lat2 < 1e-12:
        return 0.0, 0.0, (THRUST_MAX / mass) if az >= 0 else -(THRUST_MAX / mass), THRUST_MAX
    # |F|^2 = m^2 (az^2 + s^2 * a_lat2) = THRUST_MAX^2  =>  s = sqrt((T^2/m^2 - az^2) / a_lat2)
    need = a_max * a_max - az * az
    if need <= 0:
        s = THRUST_MAX / F_cmd
        return ax * s, ay * s, az * s, THRUST_MAX
    s_lat = math.sqrt(need / a_lat2)
    return ax * s_lat, ay * s_lat, az, THRUST_MAX


# ═══════════════════════════════════════════════════════════════════
#  栅格翼水平修正 (无推力时)
# ═══════════════════════════════════════════════════════════════════

def gridfin_lateral(x, y, vx, vy, vz, z, mass):
    """
    自由下落段栅格翼气动侧向修正.
    能力有限: 最大侧力 ≈ 总阻力的 5%.
    """
    v = math.sqrt(vx * vx + vy * vy + vz * vz)
    if v < 1.0 or z < 1.0:
        return 0.0, 0.0

    rho = atm_density(z)
    a_lat_max = 0.5 * rho * v * v * CD_BODY * A_REF * 0.05 / mass

    t_est = max(z / max(abs(vz), 1.0), 0.5)
    ax_des = -2.0 * x / (t_est * t_est) - 2.0 * vx / t_est
    ay_des = -2.0 * y / (t_est * t_est) - 2.0 * vy / t_est

    a_des = math.sqrt(ax_des * ax_des + ay_des * ay_des)
    if a_des > a_lat_max and a_des > 0.01:
        s = a_lat_max / a_des
        ax_des *= s
        ay_des *= s

    return ax_des, ay_des


# ═══════════════════════════════════════════════════════════════════
#  姿态模型
# ═══════════════════════════════════════════════════════════════════

def attitude_from_thrust(ax_t, ay_t, az_t):
    """
    由推力矢量方向推算欧拉角 (ZYX 约定).
      pitch = atan2(ax, az)   → +X 推力 ↔ +pitch
      roll  = −atan2(ay, az)  → +Y 推力 ↔ −roll
    """
    a_up = max(az_t, 0.1)
    pitch = math.atan2(ax_t, a_up)
    roll  = -math.atan2(ay_t, a_up)
    yaw   = 0.0
    return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════════
#  主仿真
# ═══════════════════════════════════════════════════════════════════

def generate_trajectory(out_path='trajectory.csv', verbose=True):
    """生成 Falcon 9 一级着陆段完整 6-DoF 轨迹."""

    if verbose:
        print("=" * 62)
        print("  Falcon 9 一级着陆段 6DoF 仿真轨迹生成 (理想情况)")
        print("=" * 62)

    # 状态初始化
    t  = 0.0
    x,  y,  z  = float(X0), float(Y0), float(Z0)
    vx, vy, vz = float(VX0), float(VY0), float(VZ0)
    mass = M_DRY + M_FUEL

    engine_on     = False
    t_ignition    = None
    z_ignition    = None   # 点火时刻高度 (m)，用于制导参考剖面
    vz_ignition   = None   # 点火时刻垂直速度 (m/s)
    fuel_consumed = 0.0

    # 姿态平滑滤波器状态
    TAU_ATT   = 0.15       # 姿态时间常数 (s)
    alpha_att = DT / (TAU_ATT + DT)
    roll_s, pitch_s, yaw_s = 0.0, math.radians(1.5), math.radians(-0.5)

    states = []

    while z > 0 and t < 120.0:
        # ── 气动阻力 ──
        dax, day, daz = aero_drag_accel(vx, vy, vz, z, mass)

        # ── 发动机推力 / 制导 ──
        tax, tay, taz = 0.0, 0.0, 0.0
        F_thrust = 0.0

        if not engine_on:
            # 若设置了点火高度则在该高度点火，否则用能量约束自动判据
            if Z_IGNITION_ALT is not None:
                engine_on = z <= Z_IGNITION_ALT
            else:
                engine_on = should_ignite(z, vz, z, vx, vy, mass)
            if engine_on:
                t_ignition = t
                z_ignition = z
                vz_ignition = vz
                if verbose:
                    v_now = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                    print(f"\n  [点火] t = {t:.2f} s,  z = {z:.1f} m,  "
                          f"|v| = {v_now * 3.6:.1f} km/h,  m = {mass:.0f} kg")

        if engine_on and mass > M_DRY + 50.0:
            tax, tay, taz = powered_guidance(x, y, z, vx, vy, vz,
                                             z_ignition=z_ignition, vz_ignition=vz_ignition)
            # 推力爬升: 点火后 T_THRUST_RAMP 秒内从 0 线性增至指令值，使加速度连续
            if t_ignition is not None and T_THRUST_RAMP > 0:
                ramp = min(1.0, (t - t_ignition) / T_THRUST_RAMP)
                tax, tay, taz = ramp * tax, ramp * tay, ramp * taz
            tax, tay, taz, F_thrust = apply_thrust_limits(tax, tay, taz, mass)
            if F_thrust > 0:
                dm = F_thrust / (ISP_SL * G) * DT
                mass -= dm
                fuel_consumed += dm

        # ── 栅格翼修正 (仅无推力段) ──
        gfx, gfy = 0.0, 0.0
        if not engine_on:
            gfx, gfy = gridfin_lateral(x, y, vx, vy, vz, z, mass)

        # ── 总加速度 ──
        ax = dax + tax + gfx
        ay = day + tay + gfy
        az = -G + daz + taz

        # ── 姿态 (原始值 + 一阶低通滤波) ──
        if engine_on and F_thrust > 0:
            roll_raw, pitch_raw, yaw_raw = attitude_from_thrust(tax, tay, taz)
        else:
            decay = math.exp(-t / 6.0)
            roll_raw  = 0.0
            pitch_raw = math.radians(1.5) * decay
            yaw_raw   = math.radians(-0.5) * decay
        roll_s  = alpha_att * roll_raw  + (1 - alpha_att) * roll_s
        pitch_s = alpha_att * pitch_raw + (1 - alpha_att) * pitch_s
        yaw_s   = alpha_att * yaw_raw   + (1 - alpha_att) * yaw_s
        roll, pitch, yaw = roll_s, pitch_s, yaw_s

        # ── 记录 ──
        states.append((t, x, y, z, roll, pitch, yaw,
                        vx, vy, vz, ax, ay, az))

        # ── 数值积分 (半隐式 Euler) ──
        vx += ax * DT;  vy += ay * DT;  vz += az * DT
        x  += vx * DT;  y  += vy * DT;  z  += vz * DT
        t  += DT

    # ── 终端修正: 线性插值到 z = 0 ──
    if len(states) >= 2:
        s0 = states[-2]
        s1 = states[-1]
        z0_last, z1_last = s0[3], s1[3]
        if z0_last > 0 >= z1_last and (z0_last - z1_last) > 1e-6:
            alpha = z0_last / (z0_last - z1_last)
            interp = tuple(a + alpha * (b - a) for a, b in zip(s0, s1))
            states[-1] = (interp[0],
                          interp[1], interp[2], 0.0,
                          interp[4], interp[5], interp[6],
                          interp[7], interp[8], interp[9],
                          interp[10], interp[11], interp[12])

    # ── 写出 CSV ──
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x', 'y', 'z',
                         'roll', 'pitch', 'yaw',
                         'vx', 'vy', 'vz',
                         'ax', 'ay', 'az'])
        for row in states:
            writer.writerow([
                f"{row[0]:.4f}",
                f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}",
                f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}",
                f"{row[7]:.4f}", f"{row[8]:.4f}", f"{row[9]:.4f}",
                f"{row[10]:.4f}", f"{row[11]:.4f}", f"{row[12]:.4f}",
            ])

    # ── 打印摘要 ──
    if verbose:
        ld = states[-1]
        v0_act = math.sqrt(VX0 ** 2 + VY0 ** 2 + VZ0 ** 2)
        vf = math.sqrt(ld[7] ** 2 + ld[8] ** 2 + ld[9] ** 2)
        herr = math.sqrt(ld[1] ** 2 + ld[2] ** 2)

        print(f"\n  全轨迹: {len(states)} 帧  |  仿真时长 {ld[0]:.2f} s  "
              f"|  采样率 {1 / DT:.0f} Hz")
        print(f"  初始:  z = {Z0:.0f} m,  |v| = {v0_act * 3.6:.1f} km/h")
        if t_ignition is not None:
            print(f"  点火:  t = {t_ignition:.2f} s")
        print(f"  触地:  x = {ld[1]:.3f} m,  y = {ld[2]:.3f} m,  z = {ld[3]:.3f} m")
        print(f"  触地速度: vz = {ld[9]:.3f} m/s,  |v| = {vf * 3.6:.2f} km/h")
        print(f"  水平偏差: {herr:.3f} m")
        print(f"  燃料消耗: {fuel_consumed:.1f} / {M_FUEL:.0f} kg "
              f"({100 * fuel_consumed / M_FUEL:.1f}%)")
        print(f"\n  轨迹已保存 → {out_path}")

    return out_path


# ═══════════════════════════════════════════════════════════════════
#  可视化 (matplotlib)
# ═══════════════════════════════════════════════════════════════════

def plot_trajectory(csv_path, save_path=None):
    """读取轨迹 CSV 并生成 6 子图可视化."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib / numpy 未安装, 跳过可视化")
        return None

    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    t = data['time']
    x, y, z = data['x'], data['y'], data['z']
    vx, vy, vz = data['vx'], data['vy'], data['vz']
    roll, pitch, yaw = data['roll'], data['pitch'], data['yaw']
    v_total = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle('Falcon 9 First Stage Landing Trajectory  (Ideal Case)',
                 fontsize=14, fontweight='bold')

    # 1 — 地面轨迹
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(x, y, 'royalblue', lw=1.5)
    ax1.plot(x[0], y[0], 'o', color='limegreen', ms=8, label='Start')
    ax1.plot(x[-1], y[-1], '*', color='red', ms=12, label='Landing')
    ax1.set_xlabel('X / East (m)');  ax1.set_ylabel('Y / North (m)')
    ax1.set_title('Ground Track');   ax1.legend(fontsize=8)
    ax1.grid(True, alpha=.3);       ax1.set_aspect('equal')

    # 2 — 高度-时间
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, z, 'crimson', lw=1.5)
    ax2.set_xlabel('Time (s)');  ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude vs Time');  ax2.set_xlim(0, 40);  ax2.grid(True, alpha=.3)

    # 3 — 速度-时间
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, v_total * 3.6, 'forestgreen', lw=1.5, label='|V|')
    ax3.plot(t, np.abs(vz) * 3.6, '--', color='steelblue', lw=1, alpha=.7,
             label='|Vz|')
    ax3.axhline(abs(VZ_LAND) * 3.6, ls=':', color='gray', lw=.8,
                label=f'Target {abs(VZ_LAND)*3.6:.0f} km/h')
    ax3.set_xlabel('Time (s)');  ax3.set_ylabel('Speed (km/h)')
    ax3.set_title('Speed vs Time');  ax3.set_xlim(0, 40);  ax3.legend(fontsize=8)
    ax3.grid(True, alpha=.3)

    # 4 — 速度-高度
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(z, v_total * 3.6, 'darkorchid', lw=1.5)
    ax4.set_xlabel('Altitude (m)');  ax4.set_ylabel('Speed (km/h)')
    ax4.set_title('Speed vs Altitude');  ax4.grid(True, alpha=.3)

    # 5 — 姿态角
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, np.degrees(roll),  'b', lw=1, label='Roll')
    ax5.plot(t, np.degrees(pitch), 'r', lw=1, label='Pitch')
    ax5.plot(t, np.degrees(yaw),   'g', lw=1, label='Yaw')
    ax5.set_xlabel('Time (s)');  ax5.set_ylabel('Attitude (°)')
    ax5.set_title('Attitude Angles vs Time');  ax5.set_xlim(0, 30)
    ax5.legend(fontsize=8);      ax5.grid(True, alpha=.3)

    # 6 — 3D 轨迹
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.plot(x, y, z, 'k-', lw=1.5)
    ax6.scatter(*[[x[0]], [y[0]], [z[0]]], c='limegreen', s=50)
    ax6.scatter(*[[x[-1]], [y[-1]], [z[-1]]], c='red', s=80, marker='*')
    ax6.set_xlabel('X (m)');  ax6.set_ylabel('Y (m)');  ax6.set_zlabel('Z (m)')
    ax6.set_title('3D Trajectory');  ax6.view_init(25, -60)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is None:
        save_path = csv_path.replace('.csv', '_plot.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  轨迹可视化 → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Falcon 9 一级着陆段 6DoF 仿真轨迹生成器')
    parser.add_argument('--out', default='trajectory.csv',
                        help='输出 CSV 路径 (默认: trajectory.csv)')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    args = parser.parse_args()

    csv_file = generate_trajectory(out_path=args.out, verbose=not args.quiet)

    if args.plot:
        plot_trajectory(csv_file)
