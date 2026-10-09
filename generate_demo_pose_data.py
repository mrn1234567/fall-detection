# -*- coding: utf-8 -*-
"""
人员跌倒检测 —— 生成演示用关键点数据
================================================
因为你现在还没有真实采集的数据，这个脚本会生成一份"演示数据"，
用来演示数据清洗流程能做什么。

它模拟了 MediaPipe 输出的 33 个关键点（每帧一行），包含：
    · 正常站立帧（label=正常）
    · 跌倒帧（label=跌倒，人体横向躺倒）
    · 故意混入的"脏数据"：
        - 低置信度关键点（部分点没检测到，v 很小）
        - 突变异常点（某帧某点坐标突然飞出去）
        - 低质量帧（一帧只有 2~3 个可见点）
这样清洗脚本才有东西可清，你能直观看到清洗前后对比。

【真实使用时】不要用这份模拟数据，
应该用 MediaPipe 从摄像头/视频里提取真实关键点，
格式完全一致（frame_id,label,x0..v32），即可替换。
"""

import numpy as np
import pandas as pd

NUM_POINTS = 33
OUTPUT_FILE = 'pose_data.csv'

# ---------------------------------------------------------------
# 直立人体 33 个关键点的基准坐标 (x, y)，y 向下增大（图像坐标）
# 索引对应 MediaPipe Pose 的关键点编号
# ---------------------------------------------------------------
UPRIGHT = {
    0:  (0.50, 0.18),   # 鼻
    1:  (0.53, 0.20), 2:  (0.50, 0.20), 3:  (0.47, 0.20),  # 左眼
    4:  (0.56, 0.20), 5:  (0.50, 0.20), 6:  (0.44, 0.20),  # 右眼
    7:  (0.57, 0.21), 8:  (0.43, 0.21),                    # 耳
    9:  (0.50, 0.22), 10: (0.50, 0.22),                    # 嘴
    11: (0.44, 0.33), 12: (0.56, 0.33),                    # 肩
    13: (0.40, 0.44), 14: (0.60, 0.44),                    # 肘
    15: (0.38, 0.52), 16: (0.62, 0.52),                    # 腕
    17: (0.37, 0.50), 18: (0.63, 0.50),                    # 小指
    19: (0.39, 0.50), 20: (0.61, 0.50),                    # 食指
    21: (0.40, 0.51), 22: (0.60, 0.51),                    # 拇指
    23: (0.46, 0.55), 24: (0.54, 0.55),                    # 髋
    25: (0.45, 0.75), 26: (0.55, 0.75),                    # 膝
    27: (0.44, 0.92), 28: (0.56, 0.92),                    # 踝
    29: (0.45, 0.94), 30: (0.55, 0.94),                    # 脚跟
    31: (0.43, 0.95), 32: (0.57, 0.95),                    # 脚尖
}

# ---------------------------------------------------------------
# 跌倒（平躺）姿态：身体沿 x 轴横向铺开，y 都接近同一水平线
# 头在右 (x≈0.8)，脚在左 (x≈0.27)
# ---------------------------------------------------------------
FALL_X = {
    0: 0.80, 7: 0.77, 8: 0.77, 11: 0.71, 12: 0.71,
    13: 0.66, 14: 0.66, 23: 0.55, 24: 0.55,
    25: 0.42, 26: 0.42, 27: 0.30, 28: 0.30,
    29: 0.29, 30: 0.29, 31: 0.27, 32: 0.27,
}
FALL_Y = 0.62   # 平躺时所有点都在这条水平线附近


def make_pose(mode, rng):
    """生成一帧 33 个关键点的 [x,y,z,v]，mode: 'normal' 或 'fall'。"""
    points = []
    for i in range(NUM_POINTS):
        if mode == 'normal':
            bx, by = UPRIGHT.get(i, (0.5, 0.5))
            # 加一点自然的晃动
            x = bx + rng.normal(0, 0.006)
            y = by + rng.normal(0, 0.006)
        else:
            x = FALL_X.get(i, 0.5) + rng.normal(0, 0.006)
            y = FALL_Y + rng.normal(0, 0.006)
        z = rng.normal(0, 0.05)          # 深度（不重要，给个噪声）
        v = rng.uniform(0.80, 0.99)      # 置信度，正常都较高
        points += [x, y, z, v]
    return points


def inject_dirt(rng, n_total):
    """记录要注入脏数据的帧号，返回集合。"""
    dirty_low_conf = set(rng.choice(n_total, size=int(n_total*0.06), replace=False))
    dirty_jump    = set(rng.choice(n_total, size=int(n_total*0.04), replace=False))
    dirty_badframe= set(rng.choice(n_total, size=int(n_total*0.05), replace=False))
    return dirty_low_conf, dirty_jump, dirty_badframe


def main():
    rng = np.random.default_rng(42)
    n_normal, n_fall = 400, 100
    n_total = n_normal + n_fall
    dirty_low_conf, dirty_jump, dirty_badframe = inject_dirt(rng, n_total)

    rows = []
    fi = 0
    # 先正常段，后跌倒段
    for _ in range(n_normal):
        mode = 'normal'
        label = '正常'
        pts = make_pose(mode, rng)
        # 注入低置信度：随机把 2~4 个点置信度调低
        if fi in dirty_low_conf:
            for _k in range(rng.integers(2, 5)):
                pi = int(rng.integers(0, NUM_POINTS))
                pts[pi*4+3] = rng.uniform(0.1, 0.3)
        # 注入突变：随机把某点 x,y 弹到远处
        if fi in dirty_jump:
            pi = int(rng.integers(0, NUM_POINTS))
            pts[pi*4+0] = rng.uniform(0.0, 1.0)
            pts[pi*4+1] = rng.uniform(0.0, 1.0)
        # 注入低质量帧：只保留 2 个可见点
        if fi in dirty_badframe:
            keep = rng.choice(NUM_POINTS, size=2, replace=False)
            for pi in range(NUM_POINTS):
                if pi not in keep:
                    pts[pi*4+3] = 0.05
        rows.append([fi, label] + pts)
        fi += 1

    for _ in range(n_fall):
        mode = 'fall'
        label = '跌倒'
        pts = make_pose(mode, rng)
        if fi in dirty_low_conf:
            for _k in range(rng.integers(2, 5)):
                pi = int(rng.integers(0, NUM_POINTS))
                pts[pi*4+3] = rng.uniform(0.1, 0.3)
        if fi in dirty_jump:
            pi = int(rng.integers(0, NUM_POINTS))
            pts[pi*4+0] = rng.uniform(0.0, 1.0)
            pts[pi*4+1] = rng.uniform(0.0, 1.0)
        if fi in dirty_badframe:
            keep = rng.choice(NUM_POINTS, size=2, replace=False)
            for pi in range(NUM_POINTS):
                if pi not in keep:
                    pts[pi*4+3] = 0.05
        rows.append([fi, label] + pts)
        fi += 1

    cols = ['frame_id', 'label']
    for i in range(NUM_POINTS):
        cols += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f'已生成演示数据：{OUTPUT_FILE}')
    print(f'总帧数：{len(df)}（正常 {n_normal} 帧，跌倒 {n_fall} 帧）')
    print(f'注入脏数据：低置信度 {len(dirty_low_conf)} 帧，'
          f'突变异常 {len(dirty_jump)} 帧，低质量帧 {len(dirty_badframe)} 帧')
    print('这只是一份演示数据，正式使用时请替换为 MediaPipe 真实采集数据。')


if __name__ == '__main__':
    main()
