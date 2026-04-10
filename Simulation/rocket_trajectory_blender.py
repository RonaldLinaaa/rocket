# 火箭软着陆轨迹仿真渲染脚本
# 适配版本：Blender 5.0.x
#
# 场景说明：
#   - 火箭喷口朝下，执行软着陆回收
#   - 起始点：(50, -25, 3000m)，几乎竖直下降
#   - 落点：(-1, 0.5, ~0m)，精确受控着陆
#   - 地面：沙漠地形（asset_base_id:b87cfaf0-f78d-452f-93e0-52461350c941 asset_type:model）
#   - 着陆区：圆形标志 + H 字标志（海拔 0）
#
# 使用方法：
#   1. 修改下方 BASE_DIR 为素材所在目录
#   2. 在 Blender 5.0 Scripting 工作区打开本脚本，点击 Run Script
#   3. 空格键预览，F12 渲染单帧，Ctrl+F12 渲染全序列


import bpy
import bmesh
import csv
import math
import os
from mathutils import Euler, Vector
from bpy_extras import anim_utils

# ═══════════════════════════════════════════════════════
# 0. 路径配置（按需修改）
# ═══════════════════════════════════════════════════════

BASE_DIR     = r"E:\\1_Projects\\毕设\\项目工作区"
CSV_PATH     = os.path.join(BASE_DIR, "Visible Light Visual Positioning\\Simulation\\trajectory_01.csv")
ENV_DIR      = os.path.join(BASE_DIR, "blender simulation\\eroded-ridge\\eroded-ridge.blend")  # 环境纹理目录

KEYFRAME_STEP = 1    # 每 N 行插一个关键帧
FPS           = 50

# 火箭模型尺寸（软着陆回收火箭，参考猎鹰9一子级比例）
ROCKET_RADIUS = 1.85   # 箭体半径 m（猎鹰9约1.83m）
ROCKET_LENGTH = 42.0   # 箭体长度 m

# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

def get_fcurves(obj):
    anim_data = obj.animation_data
    if not anim_data or not anim_data.action:
        return []
    action = anim_data.action
    try:
        slot = anim_data.action_slot
        channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
        if channelbag:
            return channelbag.fcurves
    except AttributeError:
        pass
    try:
        return action.fcurves
    except AttributeError:
        return []


def get_bsdf(mat):
    return next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def set_emission(bsdf, color, strength):
    for name in ("Emission Color", "Emission"):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = (color[0], color[1], color[2], 1.0)
            break
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength


def apply_all(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# ═══════════════════════════════════════════════════════
# 1. 清空场景
# ═══════════════════════════════════════════════════════

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for d in [bpy.data.meshes, bpy.data.materials, bpy.data.curves,
          bpy.data.lights, bpy.data.cameras]:
    for item in list(d):
        if item.users == 0:
            d.remove(item)


# ═══════════════════════════════════════════════════════
# 2. 读取轨迹数据
# ═══════════════════════════════════════════════════════

trajectory = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        trajectory.append({
            't':     float(row['time']),
            'x':     float(row['x']),
            'y':     float(row['y']),
            'z':     float(row['z']),
            'roll':  float(row['roll']),
            'pitch': float(row['pitch']),
            'yaw':   float(row['yaw']),
        })

total_frames = len(trajectory)          # 每行 = 一帧，总帧数 = 数据行数
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = total_frames
bpy.context.scene.render.fps  = FPS
start = trajectory[0]
end   = trajectory[-1]
mid   = trajectory[len(trajectory) // 2]
print(f"[Info] {len(trajectory)} 个数据点  |  {total_frames} 帧（每行一帧）")
print(f"[Info] 起始 z={start['z']:.0f}m  →  落点 z={end['z']:.3f}m")


# ═══════════════════════════════════════════════════════
# 3. 火箭模型
#
# 坐标约定：
#   +Z = 上（头部方向）
#   -Z = 下（喷口方向，朝向地面）
#   火箭在下降时喷口朝下喷火减速，头部朝上
#   roll/pitch/yaw 均极小（±0.07 rad），几乎竖直
# ═══════════════════════════════════════════════════════

def make_mat_metal(name, color, roughness=0.25, metallic=0.9):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = get_bsdf(mat)
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Metallic"].default_value   = metallic
    b.inputs["Roughness"].default_value  = roughness
    return mat


def create_rocket():
    R = ROCKET_RADIUS
    L = ROCKET_LENGTH
    parts = []

    # ── 主箭体（圆柱，底部 z=0，顶部 z=L）
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=R, depth=L,
                                        location=(0, 0, L * 0.5))
    body = bpy.context.active_object
    body.name = "Rocket_Body"
    apply_all(body)
    parts.append(body)

    # ── 鼻锥（头部 +Z 端，高度 L*0.18）
    nh = L * 0.18
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=R, radius2=0, depth=nh,
                                    location=(0, 0, L + nh * 0.5))
    nose = bpy.context.active_object
    nose.name = "Rocket_Nose"
    apply_all(nose)
    parts.append(nose)

    # ── 合并所有部件
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    rocket = bpy.context.active_object
    rocket.name = "Rocket"

    # ── 材质：白色金属漆
    mat = make_mat_metal("Rocket_Mat", (0.90, 0.90, 0.92))
    rocket.data.materials.append(mat)
    return rocket


rocket = create_rocket()



# ═══════════════════════════════════════════════════════
# 4. 关键帧动画
# ═══════════════════════════════════════════════════════

print("[Info] 写入关键帧...")
bpy.context.scene.frame_set(1)
rocket.rotation_mode = 'XYZ'

# 每一行数据直接对应一帧（frame = 行序号 + 1）
# 不再依赖 time 列换算，也不跳行采样
for i, pt in enumerate(trajectory):
    frame = i + 1
    rocket.location       = (pt['x'], pt['y'], pt['z'])
    rocket.rotation_euler = Euler((pt['roll'], pt['pitch'], pt['yaw']), 'XYZ')
    rocket.keyframe_insert(data_path="location",       frame=frame)
    rocket.keyframe_insert(data_path="rotation_euler", frame=frame)

# 相邻帧之间用 LINEAR 插值，忠实还原每行数据，不做额外平滑
for fc in get_fcurves(rocket):
    for kfp in fc.keyframe_points:
        kfp.interpolation = 'LINEAR'

print(f"[Info] 关键帧完成  共 {len(trajectory)} 帧")


# ═══════════════════════════════════════════════════════
# 5. 轨迹曲线（橙色发光 NURBS）
# ═══════════════════════════════════════════════════════

cd = bpy.data.curves.new("TrajCurve", 'CURVE')
cd.dimensions = '3D'
cd.bevel_depth = 0.25
sp = cd.splines.new('NURBS')
sp.points.add(len(trajectory) - 1)
for i, pt in enumerate(trajectory):
    sp.points[i].co = (pt['x'], pt['y'], pt['z'], 1.0)

traj_obj = bpy.data.objects.new("TrajCurve", cd)
bpy.context.collection.objects.link(traj_obj)

mat_traj = bpy.data.materials.new("Traj_Mat")
mat_traj.use_nodes = True
b = get_bsdf(mat_traj)
b.inputs["Base Color"].default_value = (1.0, 0.45, 0.05, 1.0)
set_emission(b, (1.0, 0.45, 0.05), 3.0)
traj_obj.data.materials.append(mat_traj)


# ═══════════════════════════════════════════════════════
# 6. 地形环境导入
#
# 从 .blend 文件链接沙漠地形 mesh 对象，
# 导入后施加变换：
#   平移  (x, y, z) = (-24m, -26m, -17m)
#   缩放  (x, y, z) = (2,    3,    1   )
# ENV_DIR 已在第 0 节配置为 .blend 文件完整路径。
# ═══════════════════════════════════════════════════════
 
print("[Info] 正在导入地形环境（3×3 拼接）...")

# 着陆点坐标（轨迹末点），提前定义供地形居中使用
LZ_X = end['x']
LZ_Y = end['y']


# ── 6.1  地形集合准备
terrain_collection_name = "Terrain"
if terrain_collection_name not in bpy.data.collections:
    terrain_col = bpy.data.collections.new(terrain_collection_name)
    bpy.context.scene.collection.children.link(terrain_col)
else:
    terrain_col = bpy.data.collections[terrain_collection_name]

# ── 6.2  读取 .blend 内对象名称列表（只读一次）
with bpy.data.libraries.load(ENV_DIR, link=False) as (data_from, _):
    _src_obj_names = list(data_from.objects)

if not _src_obj_names:
    print("[Warning] ENV_DIR 中未找到任何对象，请检查路径。")

# ── 6.3  导入第一块，获取 bounding box 尺寸，用于计算拼接步长
#         首块放在原点，后续 8 块按 tile_size 偏移后复制
def _append_tile(suffix):
    """从 .blend 追加一份地形对象，返回对象列表"""
    with bpy.data.libraries.load(ENV_DIR, link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    objs = []
    for obj in data_to.objects:
        if obj is None:
            continue
        if obj.name not in bpy.context.collection.objects:
            bpy.context.collection.objects.link(obj)
        # 重命名避免冲突
        obj.name = f"Terrain_{suffix}_{obj.name}"
        objs.append(obj)
    return objs

# 追加第一块，用于测量尺寸
_tile0_objs = _append_tile("r0c0")

# 计算第一块所有对象的合并世界 bounding box（仅取原始 scale=1 时的尺寸）
_min_x = _min_y =  1e18
_max_x = _max_y = -1e18
for _o in _tile0_objs:
    for _corner in _o.bound_box:
        wx = _o.matrix_world @ __import__('mathutils').Vector(_corner)
        _min_x = min(_min_x, wx.x); _max_x = max(_max_x, wx.x)
        _min_y = min(_min_y, wx.y); _max_y = max(_max_y, wx.y)

TILE_SIZE_X = _max_x - _min_x   # 单块地形 X 方向世界尺寸
TILE_SIZE_Y = _max_y - _min_y   # 单块地形 Y 方向尺寸
print(f"[Info] 单块地形尺寸：X={TILE_SIZE_X:.1f}m  Y={TILE_SIZE_Y:.1f}m")

# ── 6.4  追加剩余 8 块，按 3×3 网格偏移放置
#         网格原点对齐场景中心（LZ 附近），整体居中
#   行列索引 (row, col) ∈ {0,1,2}²
#   tile(1,1) 为中心块，位移 = 0
#   X 偏移 = (col-1) * TILE_SIZE_X
#   Y 偏移 = (row-1) * TILE_SIZE_Y

TERRAIN_Z_LOC   = -1.0    # z 方向整体下移 1m
TERRAIN_Z_SCALE = 0.001   # z 方向压缩，平整地形

# 3×3 网格中心块 (row=1,col=1) 对齐到 (LZ_X, LZ_Y)
TERRAIN_CENTER_X = LZ_X
TERRAIN_CENTER_Y = LZ_Y

all_terrain_objects = []

for _row in range(3):
    for _col in range(3):
        _tag = f"r{_row}c{_col}"
        if _row == 0 and _col == 0:
            # 第一块已追加，直接使用
            _tile_objs = _tile0_objs
        else:
            _tile_objs = _append_tile(_tag)

        # 计算该块的位移（中心块偏移为 0）
        _dx = (_col - 1) * TILE_SIZE_X + TERRAIN_CENTER_X
        _dy = (_row - 1) * TILE_SIZE_Y + TERRAIN_CENTER_Y

        for _obj in _tile_objs:
            # XY：只做平移，不缩放（保持原始 XY 尺寸）
            _obj.location.x += _dx
            _obj.location.y += _dy
            _obj.location.z += TERRAIN_Z_LOC
            # Z 方向压缩
            _obj.scale.z *= TERRAIN_Z_SCALE
            # 移入地形集合
            for _parent_col in list(_obj.users_collection):
                _parent_col.objects.unlink(_obj)
            terrain_col.objects.link(_obj)
            all_terrain_objects.append(_obj)

print(f"[Info] 3×3 地形拼接完成，共 {len(all_terrain_objects)} 个对象"
      f"  中心=({TERRAIN_CENTER_X:.1f},{TERRAIN_CENTER_Y:.1f})"
      f"  Z缩放={TERRAIN_Z_SCALE}  Z位移={TERRAIN_Z_LOC}m")



# ═══════════════════════════════════════════════════════
# 7. 着陆区标志（落点附近，地形对齐后 z=0）
#
# Z 层次（防止 z-fighting，间距 0.02m）：
#   凸台顶面   z = LZ_Z            ← 混凝土基座顶面，与标志物底层齐平
#   圆环 / H   z = LZ_Z + 0.02    ← 平铺薄板，统一高度，白色混凝土
#
# 凸台几何：
#   顶面半宽 BG_R，底面半宽 BG_R×1.5，厚 0.5m（向下埋入）
#
# 圆环改为平面环形（外圆 - 内圆挖空），与 H 字同高，
# 不再使用 torus（截面圆形），视觉上更像地面漆线。
# ═══════════════════════════════════════════════════════
 
LZ_X = end['x']
LZ_Y = end['y']
LZ_Z = 0.0    # 标志物基准面（贴地）
 
# ──────────────────────────────────────────
# 7.0  尺寸常量
# ──────────────────────────────────────────
RING_R_OUT = 21.5   # 圆环外径（m）
RING_R_IN  = 18.5   # 圆环内径（m），线宽 = 3m
RING_SEGS  = 128    # 圆环多边形段数（越高越圆滑）
BG_R       = RING_R_OUT + 4.0   # 凸台顶面半宽（圆环外缘再扩 4m）
 
H_H   = 18.0          # H 字高度（m）
H_T   = 3.0           # H 字笔画线宽（m）
H_W   = H_H * 0.6     # H 字总宽 ≈ 10.8m
SLAB_TH = 0.08        # 圆环 / H 字薄板厚度（m），薄而平整
 
MARK_Z = LZ_Z + 0.02  # 圆环与 H 字的统一 Z（略高于凸台顶面，避免 z-fighting）
 
 
# ──────────────────────────────────────────
# 7.1  材质工厂：程序化混凝土（可复用）
#
# make_concrete_mat(name, base_dark, base_light)
#   base_dark / base_light：ColorRamp 两端颜色
#   深色凸台：近黑  →  深灰
#   白色标志：浅灰  →  近白
# ──────────────────────────────────────────
def make_concrete_mat(name, base_dark, base_light,
                      roughness=1.0, bump_str=1.0, bump_dist=0.03):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nd = mat.node_tree.nodes
    lk = mat.node_tree.links
    nd.clear()
 
    out  = nd.new("ShaderNodeOutputMaterial"); out.location  = (700,   0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (400,   0)
    bsdf.inputs["Roughness"].default_value            = roughness
    bsdf.inputs["Specular IOR Level"].default_value   = 0.0   # 无镜面反射
 
    # 宏观噪波：混凝土块状纹路
    n_macro = nd.new("ShaderNodeTexNoise"); n_macro.location = (-520, 160)
    n_macro.inputs["Scale"].default_value      = 45.0
    n_macro.inputs["Detail"].default_value     = 10.0
    n_macro.inputs["Roughness"].default_value  = 0.80
    n_macro.inputs["Distortion"].default_value = 0.45
 
    # 微观噪波：细砂砾颗粒感
    n_micro = nd.new("ShaderNodeTexNoise"); n_micro.location = (-520, -80)
    n_micro.inputs["Scale"].default_value      = 280.0
    n_micro.inputs["Detail"].default_value     = 16.0
    n_micro.inputs["Roughness"].default_value  = 1.0
    n_micro.inputs["Distortion"].default_value = 0.08
 
    # 两层噪波相乘叠加
    mix = nd.new("ShaderNodeMixRGB"); mix.location = (-220, 60)
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.65
 
    # ColorRamp：将噪波映射到目标色调区间
    ramp = nd.new("ShaderNodeValToRGB"); ramp.location = (50, 60)
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color    = (*base_dark,  1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color    = (*base_light, 1.0)
 
    # Bump：同一噪波驱动法线，制造粗糙表面起伏
    bump = nd.new("ShaderNodeBump"); bump.location = (50, -180)
    bump.inputs["Strength"].default_value = bump_str
    bump.inputs["Distance"].default_value = bump_dist
 
    lk.new(n_macro.outputs["Fac"],   mix.inputs["Color1"])
    lk.new(n_micro.outputs["Fac"],   mix.inputs["Color2"])
    lk.new(mix.outputs["Color"],     ramp.inputs["Fac"])
    lk.new(ramp.outputs["Color"],    bsdf.inputs["Base Color"])
    lk.new(mix.outputs["Color"],     bump.inputs["Height"])
    lk.new(bump.outputs["Normal"],   bsdf.inputs["Normal"])
    lk.new(bsdf.outputs["BSDF"],     out.inputs["Surface"])
    return mat
 
# 深黑混凝土（凸台基座）
mat_base = make_concrete_mat(
    "Mat_Concrete_Base",
    base_dark  = (0.04, 0.03, 0.03),   # 近黑
    base_light = (0.18, 0.16, 0.14),   # 深灰
    roughness=1.0, bump_str=1.2, bump_dist=0.04,
)
 
# 白色混凝土（圆环 + H 字）
mat_mark = make_concrete_mat(
    "Mat_Concrete_White",
    base_dark  = (0.62, 0.60, 0.58),   # 浅灰（模拟旧漆、磨损）
    base_light = (0.90, 0.89, 0.87),   # 近白
    roughness=0.92, bump_str=0.7, bump_dist=0.02,
)
 
 
# ──────────────────────────────────────────
# 7.2  凸台基座（梯形截面，下宽上窄）
#
#   顶面：z = LZ_Z，半宽 BG_R
#   底面：z = LZ_Z - 0.5，半宽 BG_R × 1.5
# ──────────────────────────────────────────
_mesh = bpy.data.meshes.new("LZ_Base_Mesh")
_obj  = bpy.data.objects.new("LZ_Base", _mesh)
bpy.context.collection.objects.link(_obj)
bpy.context.view_layer.objects.active = _obj
 
bm = bmesh.new()
 
_HALF_T = BG_R            # 顶面半宽
_HALF_B = BG_R * 1.5      # 底面半宽（1.5 倍）
_Z_T    = LZ_Z            # 顶面 Z
_Z_B    = LZ_Z - 0.5      # 底面 Z（向下 0.5m）
 
# 构建 8 个顶点（底 4 + 顶 4）
_b = _HALF_B
_vb = [bm.verts.new(( _b,  _b, _Z_B)),
       bm.verts.new((-_b,  _b, _Z_B)),
       bm.verts.new((-_b, -_b, _Z_B)),
       bm.verts.new(( _b, -_b, _Z_B))]
_t = _HALF_T
_vt = [bm.verts.new(( _t,  _t, _Z_T)),
       bm.verts.new((-_t,  _t, _Z_T)),
       bm.verts.new((-_t, -_t, _Z_T)),
       bm.verts.new(( _t, -_t, _Z_T))]
bm.verts.ensure_lookup_table()
 
bm.faces.new([_vt[0], _vt[1], _vt[2], _vt[3]])           # 顶面（法线朝上）
bm.faces.new([_vb[3], _vb[2], _vb[1], _vb[0]])           # 底面（法线朝下）
for _i in range(4):                                        # 四侧面
    _n = (_i + 1) % 4
    bm.faces.new([_vb[_i], _vb[_n], _vt[_n], _vt[_i]])
 
bm.normal_update()
bm.to_mesh(_mesh)
bm.free()
_mesh.update()
 
_obj.location = (LZ_X, LZ_Y, 0)
_obj.data.materials.append(mat_base)
 
 
# ──────────────────────────────────────────
# 7.3  平面圆环（外圆挖内圆，bmesh 布尔环形）
#
#   用两圈顶点（外圆 + 内圆）生成环形面，
#   完全平铺，厚度 = SLAB_TH，与 H 字统一高度。
# ──────────────────────────────────────────
def make_flat_ring(name, r_out, r_in, segs, z_center, thickness, mat):
    """
    生成平面圆环薄板。
    z_center：圆环中心面 Z；底面 = z_center - thickness/2，顶面 = z_center + thickness/2
    """
    import math as _math
    _mesh = bpy.data.meshes.new(name + "_Mesh")
    _obj  = bpy.data.objects.new(name, _mesh)
    bpy.context.collection.objects.link(_obj)
 
    bm = bmesh.new()
    z_bot = z_center - thickness * 0.5
    z_top = z_center + thickness * 0.5
 
    # 生成外圆和内圆两圈顶点（顶面 + 底面，共 4 圈）
    v_out_top, v_in_top, v_out_bot, v_in_bot = [], [], [], []
    for i in range(segs):
        ang = 2 * _math.pi * i / segs
        cx, cy = _math.cos(ang), _math.sin(ang)
        v_out_top.append(bm.verts.new((LZ_X + r_out * cx, LZ_Y + r_out * cy, z_top)))
        v_in_top .append(bm.verts.new((LZ_X + r_in  * cx, LZ_Y + r_in  * cy, z_top)))
        v_out_bot.append(bm.verts.new((LZ_X + r_out * cx, LZ_Y + r_out * cy, z_bot)))
        v_in_bot .append(bm.verts.new((LZ_X + r_in  * cx, LZ_Y + r_in  * cy, z_bot)))
 
    bm.verts.ensure_lookup_table()
 
    for i in range(segs):
        n = (i + 1) % segs
        # 顶面环形四边形（外→内，逆时针 = 法线朝上）
        bm.faces.new([v_out_top[i], v_in_top[i], v_in_top[n], v_out_top[n]])
        # 底面环形四边形（顺时针 = 法线朝下）
        bm.faces.new([v_out_bot[n], v_in_bot[n], v_in_bot[i], v_out_bot[i]])
        # 外侧面
        bm.faces.new([v_out_bot[i], v_out_bot[n], v_out_top[n], v_out_top[i]])
        # 内侧面
        bm.faces.new([v_in_top[i],  v_in_top[n],  v_in_bot[n],  v_in_bot[i]])
 
    bm.normal_update()
    bm.to_mesh(_mesh)
    bm.free()
    _mesh.update()
 
    _obj.data.materials.append(mat)
    return _obj
 
ring = make_flat_ring(
    name      = "LZ_Ring",
    r_out     = RING_R_OUT,
    r_in      = RING_R_IN,
    segs      = RING_SEGS,
    z_center  = MARK_Z + SLAB_TH * 0.5,
    thickness = SLAB_TH,
    mat       = mat_mark,
)
 
 
# ──────────────────────────────────────────
# 7.4  H 字标志（三块薄板）
#
#   与圆环统一高度：中心 Z = MARK_Z + SLAB_TH/2
#   板厚 = SLAB_TH（与圆环一致，视觉统一）
#   横梁两端与竖梁内侧对齐，无重叠
# ──────────────────────────────────────────
H_Z = MARK_Z + SLAB_TH * 0.5   # 薄板几何中心 Z
 
def make_h_slab(name, sx, sy, cx, cy, cz, mat):
    """sx/sy：X/Y 方向半宽；cx/cy/cz：中心世界坐标"""
    _mesh = bpy.data.meshes.new(name + "_Mesh")
    _obj  = bpy.data.objects.new(name, _mesh)
    bpy.context.collection.objects.link(_obj)
    bm = bmesh.new()
    hz = SLAB_TH * 0.5
    coords = [
        ( sx,  sy,  hz), (-sx,  sy,  hz), (-sx, -sy,  hz), ( sx, -sy,  hz),
        ( sx,  sy, -hz), (-sx,  sy, -hz), (-sx, -sy, -hz), ( sx, -sy, -hz),
    ]
    verts = [bm.verts.new(c) for c in coords]
    bm.verts.ensure_lookup_table()
    # 6 个面（上下前后左右）
    bm.faces.new([verts[0], verts[1], verts[2], verts[3]])  # 顶
    bm.faces.new([verts[7], verts[6], verts[5], verts[4]])  # 底
    bm.faces.new([verts[0], verts[3], verts[7], verts[4]])  # 右
    bm.faces.new([verts[1], verts[0], verts[4], verts[5]])  # 前
    bm.faces.new([verts[2], verts[1], verts[5], verts[6]])  # 左
    bm.faces.new([verts[3], verts[2], verts[6], verts[7]])  # 后
    bm.normal_update()
    bm.to_mesh(_mesh); bm.free(); _mesh.update()
    _obj.location = (cx, cy, cz)
    _obj.data.materials.append(mat)
    return _obj
 
# 左竖梁
make_h_slab("LZ_H_Left",
    sx = H_T * 0.5,  sy = H_H * 0.5,
    cx = LZ_X - H_W * 0.5 + H_T * 0.5,
    cy = LZ_Y, cz = H_Z, mat = mat_mark)
 
# 右竖梁
make_h_slab("LZ_H_Right",
    sx = H_T * 0.5,  sy = H_H * 0.5,
    cx = LZ_X + H_W * 0.5 - H_T * 0.5,
    cy = LZ_Y, cz = H_Z, mat = mat_mark)
 
# 横梁（两端与竖梁内侧对齐，X 方向宽度 = H_W - 2×H_T）
make_h_slab("LZ_H_Mid",
    sx = (H_W - 2 * H_T) * 0.5,  sy = H_T * 0.5,
    cx = LZ_X, cy = LZ_Y, cz = H_Z, mat = mat_mark)


# ──────────────────────────────────────────
# 7.5  道路（横竖各一条，10km，有中央黄线 + 两侧白线）
#
# 布局：
#   纵向路（沿 Y 轴）：中心 x = LZ_X + BG_R + 60，避开标志物
#   横向路（沿 X 轴）：中心 y = LZ_Y - BG_R - 60，避开标志物
#   路面宽 12m，路肩各留 1m
#   中央黄线宽 0.3m；两侧白线各宽 0.25m，距路沿 0.5m
#   所有路面 z = 0.01（略高于地面防 z-fighting）
#   线条 z = 0.02
# ──────────────────────────────────────────

ROAD_LEN   = 10000.0   # 路长 10 km
ROAD_W     = 12.0      # 路面宽 m
ROAD_Z     = 0.01      # 路面 Z
LINE_Z     = 0.02      # 线条 Z
LINE_TH    = 0.05      # 线条板厚 m

# 纵向路中心 X（标志物右侧留出 BG_R + 60m 间距）
RD_V_X = LZ_X + BG_R + 60.0
# 横向路中心 Y（标志物南侧留出 BG_R + 600m，明显远离着陆区）
RD_H_Y = LZ_Y - BG_R - 600.0

# ── 沥青路面材质（深灰，带噪波细节）
def make_asphalt_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location  = (600, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value          = 0.95
    bsdf.inputs["Specular IOR Level"].default_value = 0.02
    noise = nd.new("ShaderNodeTexNoise"); noise.location = (-300, 0)
    noise.inputs["Scale"].default_value      = 200.0
    noise.inputs["Detail"].default_value     = 12.0
    noise.inputs["Roughness"].default_value  = 0.8
    ramp = nd.new("ShaderNodeValToRGB"); ramp.location = (0, 0)
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color    = (0.06, 0.06, 0.06, 1.0)
    ramp.color_ramp.elements[1].position = 0.8
    ramp.color_ramp.elements[1].color    = (0.18, 0.17, 0.16, 1.0)
    lk.new(noise.outputs["Fac"],  ramp.inputs["Fac"])
    lk.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    lk.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])
    return mat

# ── 道路标线材质工厂（黄/白，轻微粗糙）
def make_line_mat(name, color_rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (100, 0)
    bsdf.inputs["Base Color"].default_value         = (*color_rgb, 1.0)
    bsdf.inputs["Roughness"].default_value          = 0.75
    bsdf.inputs["Specular IOR Level"].default_value = 0.05
    lk.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

mat_asphalt   = make_asphalt_mat("Mat_Asphalt")
mat_line_yel  = make_line_mat("Mat_Line_Yellow", (0.95, 0.78, 0.05))
mat_line_wht  = make_line_mat("Mat_Line_White",  (0.92, 0.92, 0.90))

# ── 生成单条道路（路面 + 中央黄线 + 两侧白线）
def make_road(name, cx, cy, length, width, along_y):
    """
    along_y=True  → 道路沿 Y 方向延伸（纵向）
    along_y=False → 道路沿 X 方向延伸（横向）
    cx/cy：路面中心 XY
    """
    half_len = length * 0.5
    half_w   = width  * 0.5

    # 路面主体
    rm = bpy.data.meshes.new(name + "_Mesh")
    ro = bpy.data.objects.new(name, rm)
    bpy.context.collection.objects.link(ro)
    bm_r = bmesh.new()
    if along_y:
        corners = [( half_w, -half_len, ROAD_Z), (-half_w, -half_len, ROAD_Z),
                   (-half_w,  half_len, ROAD_Z), ( half_w,  half_len, ROAD_Z)]
    else:
        corners = [(-half_len, -half_w, ROAD_Z), ( half_len, -half_w, ROAD_Z),
                   ( half_len,  half_w, ROAD_Z), (-half_len,  half_w, ROAD_Z)]
    vv = [bm_r.verts.new(c) for c in corners]
    bm_r.faces.new(vv)
    bm_r.normal_update(); bm_r.to_mesh(rm); bm_r.free(); rm.update()
    ro.location = (cx, cy, 0)
    ro.data.materials.append(mat_asphalt)

    # 内部辅助：生成标线薄板
    def add_stripe(sname, offset_perp, stripe_w, mat):
        sm = bpy.data.meshes.new(sname + "_Mesh")
        so = bpy.data.objects.new(sname, sm)
        bpy.context.collection.objects.link(so)
        bm_s = bmesh.new()
        hw = stripe_w * 0.5
        ht = LINE_TH  * 0.5
        if along_y:
            sc = [( hw + offset_perp, -half_len,  LINE_Z),
                  (-hw + offset_perp, -half_len,  LINE_Z),
                  (-hw + offset_perp,  half_len,  LINE_Z),
                  ( hw + offset_perp,  half_len,  LINE_Z)]
        else:
            sc = [(-half_len,  hw + offset_perp,  LINE_Z),
                  ( half_len,  hw + offset_perp,  LINE_Z),
                  ( half_len, -hw + offset_perp,  LINE_Z),
                  (-half_len, -hw + offset_perp,  LINE_Z)]
        sv = [bm_s.verts.new(c) for c in sc]
        bm_s.faces.new(sv)
        bm_s.normal_update(); bm_s.to_mesh(sm); bm_s.free(); sm.update()
        so.location = (cx, cy, 0)
        so.data.materials.append(mat)

    # 中央黄线
    add_stripe(name + "_CenterLine", 0.0,  0.30, mat_line_yel)
    # 左侧白线（负方向侧）
    add_stripe(name + "_LeftLine",  -(half_w - 0.5), 0.25, mat_line_wht)
    # 右侧白线（正方向侧）
    add_stripe(name + "_RightLine",  (half_w - 0.5), 0.25, mat_line_wht)

make_road("Road_Vertical",   RD_V_X, LZ_Y,   ROAD_LEN, ROAD_W, along_y=True)
make_road("Road_Horizontal", LZ_X,   RD_H_Y, ROAD_LEN, ROAD_W, along_y=False)

print(f"[Info] 道路生成完成  纵向X={RD_V_X:.1f}  横向Y={RD_H_Y:.1f}")


# ──────────────────────────────────────────
# 7.6  房屋（沿纵向路东侧排列，体量较大）
#
# 布局：8 栋，沿 Y 方向排列于纵向路（RD_V_X）东侧，
#   房屋西侧墙面紧贴路缘，Y 起点确保距着陆区 >20m（安全间距）
#   高度 5~8m，宽 14m，深 20m
# ──────────────────────────────────────────

HOUSE_COUNT = 8
HOUSE_W     = 14.0   # 宽（垂直路方向，X）m
HOUSE_D     = 20.0   # 深（平行路方向，Y）m
HOUSE_GAP   = 26.0   # 中心间距 m（深 20m + 间隔 6m）

# 房屋中心 X = 纵向路右侧：路中心 + 路半宽 + 间隙 + 房屋半宽
HOUSE_ROW_X = RD_V_X + ROAD_W * 0.5 + 3.0 + HOUSE_W * 0.5

# Y 起点：着陆区北侧 20m 安全线之外开始排列
HOUSE_START_Y = LZ_Y + max(BG_R + 20.0, 20.0)

# 房屋高度序列（5~8m）
HOUSE_HEIGHTS = [6.5, 7.8, 5.5, 8.0, 6.0, 7.2, 5.8, 7.5]

# ── 外墙材质：米黄色涂抹混凝土
def make_wall_mat():
    mat = bpy.data.materials.new("Mat_HouseWall")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    noise = nd.new("ShaderNodeTexNoise"); noise.location = (-300, 100)
    noise.inputs["Scale"].default_value = 80.0; noise.inputs["Detail"].default_value = 8.0
    ramp = nd.new("ShaderNodeValToRGB"); ramp.location = (0, 100)
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color    = (0.72, 0.63, 0.50, 1.0)   # 暖灰黄
    ramp.color_ramp.elements[1].position = 0.8
    ramp.color_ramp.elements[1].color    = (0.88, 0.80, 0.66, 1.0)   # 米白
    bump = nd.new("ShaderNodeBump"); bump.location = (0, -100)
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.015
    lk.new(noise.outputs["Fac"],  ramp.inputs["Fac"])
    lk.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    lk.new(noise.outputs["Fac"],  bump.inputs["Height"])
    lk.new(bump.outputs["Normal"],bsdf.inputs["Normal"])
    lk.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])
    return mat

# ── 屋顶材质：赤褐色陶瓦
def make_roof_mat():
    mat = bpy.data.materials.new("Mat_HouseRoof")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (500, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (200, 0)
    bsdf.inputs["Base Color"].default_value = (0.55, 0.22, 0.10, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.90
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    lk.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

mat_wall = make_wall_mat()
mat_roof = make_roof_mat()

def make_house(idx, cx, cy, h):
    """生成单栋房屋：主体 box + 三棱柱屋顶"""
    hw = HOUSE_W * 0.5; hd = HOUSE_D * 0.5; roof_h = h * 0.35

    # 主体
    bm_h = bmesh.new()
    coords_b = [
        ( hw,  hd, 0), (-hw,  hd, 0), (-hw, -hd, 0), ( hw, -hd, 0),
        ( hw,  hd, h), (-hw,  hd, h), (-hw, -hd, h), ( hw, -hd, h),
    ]
    vb = [bm_h.verts.new(c) for c in coords_b]
    bm_h.verts.ensure_lookup_table()
    bm_h.faces.new([vb[0],vb[1],vb[2],vb[3]])   # 底
    bm_h.faces.new([vb[7],vb[6],vb[5],vb[4]])   # 顶
    bm_h.faces.new([vb[0],vb[4],vb[5],vb[1]])   # 前
    bm_h.faces.new([vb[2],vb[6],vb[7],vb[3]])   # 后
    bm_h.faces.new([vb[1],vb[5],vb[6],vb[2]])   # 左
    bm_h.faces.new([vb[3],vb[7],vb[4],vb[0]])   # 右
    bm_h.normal_update()
    bm_mesh = bpy.data.meshes.new(f"House_{idx:02d}_Mesh")
    bm_h.to_mesh(bm_mesh); bm_h.free(); bm_mesh.update()
    body_obj = bpy.data.objects.new(f"House_{idx:02d}", bm_mesh)
    bpy.context.collection.objects.link(body_obj)
    body_obj.location = (cx, cy, 0)
    body_obj.data.materials.append(mat_wall)

    # 屋顶（三棱柱：顶部沿 X 轴的屋脊）
    bm_r = bmesh.new()
    ridge_x = hw          # 屋脊沿 X 延伸至墙边
    coords_r = [
        ( hw,  hd, h), (-hw,  hd, h), (-hw, -hd, h), ( hw, -hd, h),   # 屋檐四角
        ( hw,  0,  h + roof_h), (-hw,  0, h + roof_h),                  # 屋脊两端
    ]
    vr = [bm_r.verts.new(c) for c in coords_r]
    bm_r.verts.ensure_lookup_table()
    bm_r.faces.new([vr[0],vr[1],vr[2],vr[3]])   # 底（与墙顶重合，内部不可见）
    bm_r.faces.new([vr[0],vr[4],vr[5],vr[1]])   # 前坡
    bm_r.faces.new([vr[3],vr[2],vr[5],vr[4]])   # 后坡
    bm_r.faces.new([vr[1],vr[5],vr[2]])          # 左山墙
    bm_r.faces.new([vr[0],vr[3],vr[4]])          # 右山墙（注意绕向）
    bm_r.normal_update()
    rm_mesh = bpy.data.meshes.new(f"House_{idx:02d}_Roof_Mesh")
    bm_r.to_mesh(rm_mesh); bm_r.free(); rm_mesh.update()
    roof_obj = bpy.data.objects.new(f"House_{idx:02d}_Roof", rm_mesh)
    bpy.context.collection.objects.link(roof_obj)
    roof_obj.location = (cx, cy, 0)
    roof_obj.data.materials.append(mat_roof)

start_x = LZ_X - (HOUSE_COUNT - 1) * HOUSE_GAP * 0.5
for i in range(HOUSE_COUNT):
    hy = HOUSE_START_Y + i * HOUSE_GAP
    make_house(i, HOUSE_ROW_X, hy, HOUSE_HEIGHTS[i % len(HOUSE_HEIGHTS)])

print(f"[Info] 房屋生成完成  共 {HOUSE_COUNT} 栋  X={HOUSE_ROW_X:.1f}  Y起点={HOUSE_START_Y:.1f}")


# ──────────────────────────────────────────
# 7.7  储罐（白色，高 15m，紧接房屋列继续向北，同样沿纵向路东侧）
#
# 布局：4 个罐沿 Y 方向排列，X = 纵向路右侧（与房屋同侧）
#   Y 起点 = 房屋末端 + 间距 20m
#   2 个直径 14m（半径 7m），2 个直径 9m（半径 4.5m）
#   罐间距 = 各自直径 + 6m
# ──────────────────────────────────────────

TANK_H      = 15.0
TANK_Z_BASE = 0.0

# 储罐 X 中心与房屋同列
TANK_ROW_X = HOUSE_ROW_X

# Y 起点：房屋列末端再留 20m
TANK_START_Y = HOUSE_START_Y + (HOUSE_COUNT - 1) * HOUSE_GAP + HOUSE_D * 0.5 + 20.0

# 储罐规格（直径放大）：(radius, label)
_tank_specs = [
    (7.0,  "大罐1"),   # 直径 14m
    (7.0,  "大罐2"),   # 直径 14m
    (4.5,  "小罐1"),   # 直径  9m
    (4.5,  "小罐2"),   # 直径  9m
]

# 计算各罐 Y 中心（罐间距 = 前罐半径 + 6m + 当前罐半径）
_tank_positions = []
_cur_y = TANK_START_Y
for j, (r, _label) in enumerate(_tank_specs):
    if j == 0:
        _cur_y += r
    else:
        _prev_r = _tank_specs[j-1][0]
        _cur_y += _prev_r + 6.0 + r
    _tank_positions.append((r, TANK_ROW_X, _cur_y))

# ── 白色金属储罐材质（亮白，略带金属感）
def make_tank_mat():
    mat = bpy.data.materials.new("Mat_Tank_White")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value          = 0.25
    bsdf.inputs["Metallic"].default_value           = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    # 轻微脏污噪波
    noise = nd.new("ShaderNodeTexNoise"); noise.location = (-300, 100)
    noise.inputs["Scale"].default_value = 60.0; noise.inputs["Detail"].default_value = 6.0
    ramp = nd.new("ShaderNodeValToRGB"); ramp.location = (0, 100)
    ramp.color_ramp.elements[0].position = 0.4
    ramp.color_ramp.elements[0].color    = (0.82, 0.82, 0.82, 1.0)
    ramp.color_ramp.elements[1].position = 0.9
    ramp.color_ramp.elements[1].color    = (0.96, 0.96, 0.96, 1.0)
    lk.new(noise.outputs["Fac"],  ramp.inputs["Fac"])
    lk.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    lk.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])
    return mat

# ── 储罐顶盖材质（深灰，区分罐顶）
def make_tank_cap_mat():
    mat = bpy.data.materials.new("Mat_Tank_Cap")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (100, 0)
    bsdf.inputs["Base Color"].default_value = (0.25, 0.25, 0.25, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.6
    bsdf.inputs["Metallic"].default_value   = 0.8
    lk.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

mat_tank     = make_tank_mat()
mat_tank_cap = make_tank_cap_mat()

def make_tank(name, cx, cy, radius, height, segs=48):
    """圆柱主体 + 半球形顶盖"""
    # 主体圆柱
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segs, radius=radius, depth=height,
        location=(cx, cy, TANK_Z_BASE + height * 0.5)
    )
    body = bpy.context.active_object
    body.name = name + "_Body"
    body.data.materials.append(mat_tank)

    # 顶盖（半球，radius = 罐径，压扁为 radius*0.25 高度）
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segs, ring_count=16, radius=radius,
        location=(cx, cy, TANK_Z_BASE + height)
    )
    cap = bpy.context.active_object
    cap.name = name + "_Cap"
    cap.scale.z = 0.25   # 压扁成锅盖形
    apply_all(cap)
    cap.data.materials.append(mat_tank_cap)

for i, (r, tx, ty) in enumerate(_tank_positions):
    make_tank(f"Tank_{i+1:02d}", tx, ty, r, TANK_H)

print(f"[Info] 储罐生成完成  共 {len(_tank_positions)} 个  X={TANK_ROW_X:.1f}  Y起点={TANK_START_Y:.1f}")


# ──────────────────────────────────────────
# 7.8  长条状仓库（沿横向路北侧排列，工业感）
#
# 布局：5 栋，沿 X 方向排列于横向路（RD_H_Y）北侧，
#   每栋 60m 长 × 20m 宽 × 8m 高，间距 10m
#   屋顶为弧形（半圆柱），模拟彩钢瓦拱形仓库
#   距着陆区 > 20m（横向路本身已远离 600m 以上）
# ──────────────────────────────────────────

WH_COUNT  = 5
WH_L      = 60.0    # 长（X 方向）m
WH_W      = 20.0    # 宽（Y 方向）m
WH_H      = 8.0     # 墙高 m
WH_GAP    = 10.0    # 间距 m
WH_ROW_Y  = RD_H_Y + ROAD_W * 0.5 + 5.0 + WH_W * 0.5   # 路北侧，留 5m 间隙

# 仓库总 X 宽度居中于 LZ_X
WH_START_X = LZ_X - (WH_COUNT * WH_L + (WH_COUNT - 1) * WH_GAP) * 0.5

# ── 彩钢板墙面材质（浅蓝灰金属）
def make_warehouse_wall_mat():
    mat = bpy.data.materials.new("Mat_WH_Wall")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value          = 0.55
    bsdf.inputs["Metallic"].default_value           = 0.7
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    # 竖向条纹噪波模拟彩钢板波纹
    noise = nd.new("ShaderNodeTexNoise"); noise.location = (-400, 80)
    noise.inputs["Scale"].default_value     = 5.0
    noise.inputs["Detail"].default_value    = 2.0
    noise.inputs["Roughness"].default_value = 0.1
    ramp = nd.new("ShaderNodeValToRGB"); ramp.location = (-100, 80)
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color    = (0.45, 0.55, 0.65, 1.0)   # 钢蓝
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color    = (0.62, 0.70, 0.78, 1.0)   # 亮钢蓝
    lk.new(noise.outputs["Fac"],  ramp.inputs["Fac"])
    lk.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    lk.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])
    return mat

# ── 弧形屋顶材质（深灰金属，略氧化）
def make_warehouse_roof_mat():
    mat = bpy.data.materials.new("Mat_WH_Roof")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (500, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (200, 0)
    bsdf.inputs["Base Color"].default_value         = (0.30, 0.33, 0.35, 1.0)
    bsdf.inputs["Roughness"].default_value          = 0.70
    bsdf.inputs["Metallic"].default_value           = 0.65
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    lk.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

mat_wh_wall = make_warehouse_wall_mat()
mat_wh_roof = make_warehouse_roof_mat()

def make_warehouse(name, cx, cy):
    """长条形仓库：box 墙体 + 半圆柱拱形屋顶"""
    hl = WH_L * 0.5; hw = WH_W * 0.5

    # 墙体 box
    bm_w = bmesh.new()
    wc = [( hl,  hw, 0), (-hl,  hw, 0), (-hl, -hw, 0), ( hl, -hw, 0),
          ( hl,  hw, WH_H), (-hl,  hw, WH_H), (-hl, -hw, WH_H), ( hl, -hw, WH_H)]
    wv = [bm_w.verts.new(c) for c in wc]
    bm_w.verts.ensure_lookup_table()
    bm_w.faces.new([wv[0],wv[1],wv[2],wv[3]])
    bm_w.faces.new([wv[7],wv[6],wv[5],wv[4]])
    bm_w.faces.new([wv[0],wv[4],wv[5],wv[1]])
    bm_w.faces.new([wv[2],wv[6],wv[7],wv[3]])
    bm_w.faces.new([wv[1],wv[5],wv[6],wv[2]])
    bm_w.faces.new([wv[3],wv[7],wv[4],wv[0]])
    bm_w.normal_update()
    wm = bpy.data.meshes.new(name + "_Wall_Mesh")
    bm_w.to_mesh(wm); bm_w.free(); wm.update()
    wo = bpy.data.objects.new(name + "_Wall", wm)
    bpy.context.collection.objects.link(wo)
    wo.location = (cx, cy, 0)
    wo.data.materials.append(mat_wh_wall)

    # 拱形屋顶（半圆柱，沿 X 轴方向）
    ARCH_SEGS = 24
    ARCH_R    = WH_W * 0.5   # 半径 = 仓库半宽
    bm_r = bmesh.new()
    # 顶部弧形顶点（半圆 = 0~pi，从 -Y 到 +Y）
    arc_top = []
    arc_bot = []
    for s in range(ARCH_SEGS + 1):
        ang = math.pi * s / ARCH_SEGS   # 0 → π
        ay = -math.cos(ang) * ARCH_R    # -R → +R
        az =  math.sin(ang) * ARCH_R    # 0 → peak → 0
        arc_top.append(bm_r.verts.new(( hl, ay, WH_H + az)))
        arc_bot.append(bm_r.verts.new((-hl, ay, WH_H + az)))
    bm_r.verts.ensure_lookup_table()
    # 弧面四边形
    for s in range(ARCH_SEGS):
        bm_r.faces.new([arc_top[s], arc_top[s+1], arc_bot[s+1], arc_bot[s]])
    # 两端封口（扇形面）
    for end_verts in [arc_top, arc_bot]:
        fan = [bm_r.verts.new((end_verts[0].co.x, 0, WH_H))]  # 中心点
        face_verts = fan + end_verts[:ARCH_SEGS+1]
        # 逐三角扇形
        for s in range(ARCH_SEGS):
            bm_r.faces.new([fan[0], end_verts[s], end_verts[s+1]])
    bm_r.normal_update()
    rm = bpy.data.meshes.new(name + "_Roof_Mesh")
    bm_r.to_mesh(rm); bm_r.free(); rm.update()
    ro = bpy.data.objects.new(name + "_Roof", rm)
    bpy.context.collection.objects.link(ro)
    ro.location = (cx, cy, 0)
    ro.data.materials.append(mat_wh_roof)

for wi in range(WH_COUNT):
    wx = WH_START_X + wi * (WH_L + WH_GAP) + WH_L * 0.5
    make_warehouse(f"Warehouse_{wi+1:02d}", wx, WH_ROW_Y)

print(f"[Info] 仓库生成完成  共 {WH_COUNT} 栋  Y={WH_ROW_Y:.1f}")


# ──────────────────────────────────────────
# 7.9  灌木丛（随机分布于场景各处）
#
# 采用确定性伪随机（种子固定），避免每次运行结果不同。
# 每棵灌木 = 多个不规则椭球叠加，程序化绿色材质。
# 分布规则：
#   - 避开着陆区 (LZ_X, LZ_Y) 周围 120m
#   - 避开道路中心线 ±15m
#   - 在 X ∈ [-400, 400]，Y ∈ [-400, 400] 范围散布
#   共 60 丛
# ──────────────────────────────────────────

BUSH_COUNT   = 60
BUSH_EXCL_R  = 120.0   # 着陆区排除半径
ROAD_EXCL_W  = 15.0    # 道路排除半宽
BUSH_AREA    = 400.0   # 散布范围半径（方形）

# 灌木材质（深绿，粗糙漫反射，带噪波色调变化）
def make_bush_mat():
    mat = bpy.data.materials.new("Mat_Bush")
    mat.use_nodes = True
    nd = mat.node_tree.nodes; lk = mat.node_tree.links; nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    bsdf = nd.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value          = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    # 噪波驱动绿色色调变化（深绿~黄绿，模拟叶片密度差异）
    noise = nd.new("ShaderNodeTexNoise"); noise.location = (-300, 80)
    noise.inputs["Scale"].default_value     = 8.0
    noise.inputs["Detail"].default_value    = 6.0
    noise.inputs["Roughness"].default_value = 0.7
    ramp = nd.new("ShaderNodeValToRGB"); ramp.location = (0, 80)
    ramp.color_ramp.elements[0].position = 0.2
    ramp.color_ramp.elements[0].color    = (0.08, 0.18, 0.05, 1.0)   # 深墨绿
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color    = (0.22, 0.38, 0.10, 1.0)   # 黄绿
    # SSS 近似叶片透光感（Subsurface 输入）
    try:
        bsdf.inputs["Subsurface Weight"].default_value = 0.08
        bsdf.inputs["Subsurface Radius"].default_value = (0.05, 0.15, 0.02)
    except KeyError:
        pass
    lk.new(noise.outputs["Fac"],  ramp.inputs["Fac"])
    lk.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    lk.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])
    return mat

mat_bush = make_bush_mat()

def make_bush(name, cx, cy, scale_xy, scale_z):
    """
    单丛灌木：3~5 个椭球体随机偏移叠加。
    scale_xy: 水平尺寸基准（m），scale_z: 高度基准（m）
    使用确定性偏移（基于索引），不依赖 random 模块。
    """
    # 固定偏移模式（最多 5 个球，偏移量用简单数列）
    offsets = [
        ( 0.0,       0.0,      0.0,       1.0,  1.0 ),
        ( scale_xy * 0.4,  scale_xy * 0.2,  0.0, 0.75, 0.8),
        (-scale_xy * 0.3,  scale_xy * 0.35, 0.0, 0.65, 0.75),
        ( scale_xy * 0.1, -scale_xy * 0.4,  0.0, 0.70, 0.85),
        (-scale_xy * 0.2, -scale_xy * 0.15, scale_z * 0.3, 0.55, 0.65),
    ]
    parts = []
    for oi, (ox, oy, oz, sr_xy, sr_z) in enumerate(offsets):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=10, ring_count=8,
            radius=scale_xy * sr_xy * 0.5,
            location=(cx + ox, cy + oy, scale_z * sr_z * 0.5 + oz)
        )
        sp = bpy.context.active_object
        sp.name = f"{name}_S{oi}"
        sp.scale.z = scale_z / scale_xy * sr_z
        apply_all(sp)
        sp.data.materials.append(mat_bush)
        parts.append(sp)
    # 合并为单对象
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    bush_obj = bpy.context.active_object
    bush_obj.name = name
    return bush_obj

# 伪随机位置生成（LCG，种子固定）
def _lcg(seed):
    a, c, m = 1664525, 1013904223, 2**32
    return (a * seed + c) % m

_seed = 42
_placed = 0
_attempt = 0
while _placed < BUSH_COUNT and _attempt < BUSH_COUNT * 20:
    _attempt += 1
    _seed = _lcg(_seed);  bx = (_seed / 2**32 * 2 - 1) * BUSH_AREA
    _seed = _lcg(_seed);  by = (_seed / 2**32 * 2 - 1) * BUSH_AREA
    _seed = _lcg(_seed);  bsxy = 1.5 + (_seed / 2**32) * 2.5   # 水平 1.5~4m
    _seed = _lcg(_seed);  bsz  = 1.0 + (_seed / 2**32) * 1.5   # 高度 1~2.5m

    # 排除着陆区
    dist_lz = math.sqrt((bx - LZ_X)**2 + (by - LZ_Y)**2)
    if dist_lz < BUSH_EXCL_R:
        continue
    # 排除纵向路走廊
    if abs(bx - RD_V_X) < ROAD_EXCL_W:
        continue
    # 排除横向路走廊
    if abs(by - RD_H_Y) < ROAD_EXCL_W:
        continue

    make_bush(f"Bush_{_placed+1:03d}", bx, by, bsxy, bsz)
    _placed += 1

print(f"[Info] 灌木丛生成完成  共 {_placed} 丛（{_attempt} 次尝试）")


# ═══════════════════════════════════════════════════════
# 8. 光照
# ═══════════════════════════════════════════════════════

# 太阳光（低角度，模拟沙漠午后斜照）
bpy.ops.object.light_add(type='SUN', location=(500, -500, 4000))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy      = 4.5
sun.data.angle       = math.radians(1.5)
sun.rotation_euler   = (math.radians(55), 0, math.radians(45))

# 世界天空（简化 Preetham 大气）
world = bpy.context.scene.world
if not world:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
for n in list(wn):
    wn.remove(n)

w_out = wn.new('ShaderNodeOutputWorld');  w_out.location = (400, 0)
sky   = wn.new('ShaderNodeTexSky');       sky.location   = (0, 0)
sky.sky_type      = 'PREETHAM'
sky.sun_elevation = math.radians(45)
sky.sun_rotation  = math.radians(45)
sky.turbidity     = 2.0

bg = wn.new('ShaderNodeBackground');  bg.location = (200, 0)
bg.inputs["Strength"].default_value = 1.2
wl.new(sky.outputs["Color"],     bg.inputs["Color"])
wl.new(bg.outputs["Background"], w_out.inputs["Surface"])


# ═══════════════════════════════════════════════════════
# 9. 机载摄像机（箭体侧面，俯视地面）
# ═══════════════════════════════════════════════════════

R_cam = ROCKET_RADIUS
L_cam = ROCKET_LENGTH

# 先创建相机
bpy.ops.object.camera_add()
cam = bpy.context.active_object
cam.name = "OnboardCam"

cam.data.sensor_width = 36
cam.data.sensor_fit = 'HORIZONTAL'

# 先绑定父物体（关键）
cam.parent = rocket

# 设置局部坐标
cam.location = (
    R_cam + 0.15,     # 侧面
    0.0,
    L_cam * 0.85      # 上部
)

# 相机参数
cam.data.lens = 24
cam.data.clip_end = 6000

# 相机方向：
# Blender camera 默认 -Z 为视线
# rocket -Z = 地面方向
# 因此只需轻微向火箭中心偏转
cam.rotation_mode = 'XYZ'

cam.rotation_euler = (
    math.radians(15),0,math.radians(-90)
)

bpy.context.scene.camera = cam

print(
    f"[Info] 机载摄像机安装完成："
    f"local=({R_cam+0.15:.2f},0,{L_cam*0.85:.2f}) "
    f"视线=rocket -Z"
)


# ═══════════════════════════════════════════════════════
# 10. 渲染参数
# ═══════════════════════════════════════════════════════

scene = bpy.context.scene
scene.render.engine                     = 'CYCLES'
scene.cycles.samples                    = 128
scene.cycles.use_denoising              = True
scene.render.resolution_x               = 1080
scene.render.resolution_y               = 720
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath                   = "//rocket_render//"
scene.render.use_motion_blur            = True
scene.render.motion_blur_shutter        = 0.2

print("=" * 60)
print("[OK] 场景构建完成（Blender 5.0）")
print(f"  轨迹帧：1 ~ {total_frames} @ {FPS}fps")
print(f"  着陆区中心：({LZ_X:.2f}, {LZ_Y:.2f}, 0)")
print()
print("  空格键     → 播放预览")
print("  Numpad 0   → 摄像机视角")
print("  F12        → 渲染当前帧")
print("  Ctrl+F12   → 渲染全序列 → //rocket_render_####.png")
print("=" * 60)