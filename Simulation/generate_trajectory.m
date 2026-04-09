%% =========================================================
%  Module 1 — Rocket Descent Trajectory Generation (MATLAB)
%  可复用火箭成像引导返回段仿真轨迹生成
%
%  研究背景: 可复用火箭成像引导返回段落点精确定位方法研究
%  作者: 林航 (22374069), 宇航学院, 飞行器控制与信息工程
%  指导教师: 孟偲
%
%  仿真三阶段下降轨迹:
%    Phase 1 (3000~2000m): 高速弹道下降
%    Phase 2 (2000~800m):  姿态稳定段
%    Phase 3 (800~0m):     着陆制动燃烧段
%
%  输出: trajectory.csv (50Hz, 含6DoF状态量)
%% =========================================================

clc; clear; close all;

%% ---- 仿真参数 ----
dt = 0.02;          % 采样间隔 50Hz
g  = 9.81;          % 重力加速度 (m/s²)

% 着陆坐标系原点 = 目标着陆点 (0,0,0)
% X: East  Y: North  Z: Up
% 横向初始偏移 (模拟工程散布)
x0 =  15.0;   % m  East偏移
y0 = -10.0;   % m  North偏移
z0 = 3000.0;  % m  初始高度

% 初始速度
vx0 =  2.5;   % m/s  小幅横漂
vy0 = -1.0;   % m/s
vz0 = -118.0; % m/s  初始下降速度

% 初始姿态 (度→弧度)
roll0  = deg2rad(0.3);
pitch0 = deg2rad(1.5);  % 小幅倾斜
yaw0   = deg2rad(5.0);  % 初始偏航

%% ---- 轨迹分段生成 ----

% --- Phase 1: 3000m ~ 2000m (高速弹道段) ---
z_p1_end = 2000;
% 纵向: 重力 + 小阻力
% vz(t) = vz0 - (g - drag)*t, 加速度 ≈ -1.5 m/s² (弱阻力)
az_p1 = -1.2;  % m/s²
t_p1 = 0:dt:1000;  % 足够长,按高度截断
states_p1 = simulate_phase(t_p1, x0, y0, z0, vx0, vy0, vz0, ...
    roll0, pitch0, yaw0, ...
    0, 0, az_p1, ...          % ax, ay, az
    0, 0, 0, ...              % d_roll, d_pitch, d_yaw (rad/s)
    z_p1_end, dt);

% 取Phase1终态作为Phase2初态
s = states_p1(end,:);
x1=s(2); y1=s(3); z1=s(4);
vx1=s(8); vy1=s(9); vz1=s(10);
r1=s(5); p1=s(6); yw1=s(7);

% --- Phase 2: 2000m ~ 800m (姿态稳定段) ---
z_p2_end = 800;
% 主发动机部分节流，vz 缓慢减速; 横向修正
az_p2 = 4.0;   % 制动加速度 (向上,减小|vz|)
ax_p2 = -vx1 * 0.015;   % 横向慢修正
ay_p2 = -vy1 * 0.015;
% 姿态趋稳
dpitch = -p1 * 0.02;   % rad/s
dyaw   = -yw1 * 0.02;
t_p2 = 0:dt:1000;
states_p2 = simulate_phase(t_p2, x1, y1, z1, vx1, vy1, vz1, ...
    r1, p1, yw1, ...
    ax_p2, ay_p2, az_p2, ...
    0, dpitch, dyaw, ...
    z_p2_end, dt);

s = states_p2(end,:);
x2=s(2); y2=s(3); z2=s(4);
vx2=s(8); vy2=s(9); vz2=s(10);
r2=s(5); p2=s(6); yw2=s(7);

% --- Phase 3: 800m ~ 0m (着陆制动燃烧段) ---
z_p3_end = 0;
% 目标: 触地速度 vz_f ≈ -2 m/s, 横向趋零
% 匀减速: a_req = (vz_f² - vz2²) / (2*Δz)
delta_z = 0 - z2;   % 负值
vz_f = -2.0;
az_p3 = (vz_f^2 - vz2^2) / (2*delta_z);  % 正值(向上制动)
ax_p3 = -vx2 * 0.05;
ay_p3 = -vy2 * 0.05;
dpitch3 = -p2 * 0.05;
dyaw3   = -yw2 * 0.05;
t_p3 = 0:dt:1000;
states_p3 = simulate_phase(t_p3, x2, y2, z2, vx2, vy2, vz2, ...
    r2, p2, yw2, ...
    ax_p3, ay_p3, az_p3, ...
    0, dpitch3, dyaw3, ...
    z_p3_end, dt);

%% ---- 拼接全轨迹并构建时间轴 ----
all_states = [states_p1; states_p2; states_p3];
N = size(all_states,1);
time = (0:N-1)' * dt;

% 列顺序: [local_t, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az]
trajectory = [time, all_states(:,2:end)];

%% ---- 写出 CSV ----
headers = 'time,x,y,z,roll,pitch,yaw,vx,vy,vz,ax,ay,az';
out_file = 'trajectory.csv';
fid = fopen(out_file, 'w');
fprintf(fid, '%s\n', headers);
fclose(fid);

writematrix(trajectory, out_file, 'WriteMode', 'append');
fprintf('[INFO] 轨迹已保存至 %s\n', out_file);
fprintf('[INFO] 总帧数: %d,  仿真时长: %.1f s\n', N, time(end));
fprintf('[INFO] 触地速度: vz=%.2f m/s, vx=%.2f m/s, vy=%.2f m/s\n', ...
    trajectory(end,10), trajectory(end,8), trajectory(end,9));
fprintf('[INFO] 触地横向偏差: x=%.2f m, y=%.2f m\n', ...
    trajectory(end,2), trajectory(end,3));

%% ---- 可视化 ----
figure('Name','Rocket Descent Trajectory','Position',[100 100 1400 800]);

subplot(2,3,1);
plot(trajectory(:,2), trajectory(:,3), 'b-', 'LineWidth',1.5); grid on;
xlabel('X (m, East)'); ylabel('Y (m, North)');
title('地面轨迹投影'); axis equal;

subplot(2,3,2);
plot(trajectory(:,1), trajectory(:,4), 'r-', 'LineWidth',1.5); grid on;
xlabel('时间 (s)'); ylabel('高度 Z (m)');
title('高度-时间曲线');

subplot(2,3,3);
plot(trajectory(:,1), trajectory(:,10), 'g-', 'LineWidth',1.5); grid on;
xlabel('时间 (s)'); ylabel('Vz (m/s)');
title('垂直速度-时间曲线');
yline(-2, 'r--', '目标触地速度');

subplot(2,3,4);
plot(trajectory(:,4), trajectory(:,10), 'm-', 'LineWidth',1.5); grid on;
xlabel('高度 Z (m)'); ylabel('Vz (m/s)');
title('速度-高度关系');

subplot(2,3,5);
plot(trajectory(:,1), rad2deg(trajectory(:,5)), 'b-'); hold on;
plot(trajectory(:,1), rad2deg(trajectory(:,6)), 'r-');
plot(trajectory(:,1), rad2deg(trajectory(:,7)), 'g-'); grid on;
xlabel('时间 (s)'); ylabel('姿态角 (°)');
legend('Roll','Pitch','Yaw'); title('姿态角-时间曲线');

subplot(2,3,6);
plot3(trajectory(:,2), trajectory(:,3), trajectory(:,4), 'k-', 'LineWidth',1.5);
grid on; xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('三维下降轨迹'); view(30,30);

sgtitle('可复用火箭返回段6DoF仿真轨迹', 'FontSize',14, 'FontWeight','bold');
saveas(gcf, 'trajectory_plot.png');
fprintf('[INFO] 轨迹图已保存至 trajectory_plot.png\n');

%% =========================================================
%  辅助函数: simulate_phase
%  在单一匀加速假设下积分一段轨迹,遇 z<=z_end 截止
%% =========================================================
function states = simulate_phase(t_vec, x0, y0, z0, vx0, vy0, vz0, ...
    roll0, pitch0, yaw0, ...
    ax, ay, az, ...
    droll, dpitch, dyaw, ...
    z_end, dt)
% 返回: [t_local, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az]
    N_max = length(t_vec);
    states = zeros(N_max, 13);

    x=x0; y=y0; z=z0;
    vx=vx0; vy=vy0; vz=vz0;
    roll=roll0; pitch=pitch0; yaw=yaw0;

    for i = 1:N_max
        t_local = (i-1)*dt;
        states(i,:) = [t_local, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az];

        % 检测到达终止高度
        if z <= z_end
            states = states(1:i,:);
            return;
        end

        % 积分
        vx = vx + ax*dt;
        vy = vy + ay*dt;
        vz = vz + az*dt;
        x  = x  + vx*dt;
        y  = y  + vy*dt;
        z  = z  + vz*dt;

        roll  = roll  + droll*dt;
        pitch = pitch + dpitch*dt;
        yaw   = yaw   + dyaw*dt;
    end
end
