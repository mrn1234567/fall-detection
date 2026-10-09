# -*- coding: utf-8 -*-
"""
人员跌倒检测 —— 关键点数据清洗脚本
================================================
作用：清洗 MediaPipe 提取的人体 33 关键点数据（每帧一行），
     把脏数据弄干净，供后续训练跌倒分类模型使用。

清洗顺序（顺序很重要，尤其是"先剔异常、再归一化"）：
    ① 缺失值处理：置信度太低的关键点坐标标记为缺失(NaN)
    ② 低质量帧剔除：一帧可见关键点太少，整帧丢掉
    ③ 平滑去噪：对每个关键点坐标做滑动平均，抹平抖动
    ④ 突变异常剔除：某关键点坐标相邻帧突然飞走，剔除该点
    ⑤ 归一化：以骨盆为原点、以髋宽为尺度（放在异常剔除之后，
       否则异常值会被归一化放大）
    ⑥ 标签统计：检查各类样本数量是否平衡

依赖：pandas, numpy
安装：pip install pandas numpy

使用方法：
    1) 把原始关键点 CSV 放到本文件同目录，命名 pose_data.csv
    2) 运行：python clean_pose_data.py
    3) 清洗结果输出为 cleaned_pose_data.csv
"""

import pandas as pd
import numpy as np

# ================================================
# 可调参数（按需修改）
# ================================================
INPUT_FILE = 'pose_data.csv'          # 原始关键点数据文件名
OUTPUT_FILE = 'cleaned_pose_data.csv' # 清洗后输出文件名

VIS_TH = 0.5          # ① 关键点置信度阈值：低于它视为没检测到
MIN_VIS_POINTS = 8    # ② 一帧至少要有多少可见关键点，否则整帧删
SMOOTH_WIN = 5        # ③ 平滑窗口（奇数），越大越平滑但越迟钝
Z_TH = 3.0            # ④ 突变阈值：位移超过 3 倍标准差视为异常

NUM_POINTS = 33       # MediaPipe 人体关键点数量
# ================================================

def load_data():
    """读取原始关键点 CSV。"""
    df = pd.read_csv(INPUT_FILE)
    if 'x32' not in df.columns:
        raise ValueError(
            '未找到 x32 列，请确认文件是 MediaPipe 33 关键点数据，'
            '格式应为 frame_id,label,x0,y0,z0,v0,...,x32,y32,z32,v32')
    return df


def step1_missing(df):
    """① 缺失值处理：置信度低于阈值的坐标标记为缺失(NaN)。"""
    marked = 0
    for i in range(NUM_POINTS):
        v = df[f'v{i}']
        for c in [f'x{i}', f'y{i}', f'z{i}']:
            mask = v < VIS_TH
            df.loc[mask, c] = np.nan
            marked += int(mask.sum())
    print(f'① 缺失值处理：把 {marked} 个低置信度坐标标记为缺失(NaN)，'
          f'训练时会忽略这些点')
    return df


def step2_drop_bad_frames(df):
    """② 低质量帧剔除：可见关键点太少（如只露出一只手）整帧删掉。"""
    vis_count = np.zeros(len(df), dtype=int)
    for i in range(NUM_POINTS):
        vis_count += (df[f'v{i}'] >= VIS_TH).astype(int).values
    before = len(df)
    df = df[vis_count >= MIN_VIS_POINTS].reset_index(drop=True)
    print(f'② 低质量帧剔除：删除 {before - len(df)} 帧（{before} → {len(df)}）')
    return df


def step3_smooth(df):
    """③ 平滑去噪：滑动平均抹平抖动，打印平滑前后对比(同尺度)。"""
    # 记录平滑前的肩部相邻帧位移(取 y11，仅用于展示平滑效果)
    jitter_before = df['y11'].diff().abs().mean()
    coord_cols = [f'{c}{i}' for i in range(NUM_POINTS) for c in 'xyz']
    df[coord_cols] = df[coord_cols].rolling(
        SMOOTH_WIN, center=True, min_periods=1).mean()
    jitter_after = df['y11'].diff().abs().mean()
    print(f'③ 平滑去噪：肩部抖动 {jitter_before:.5f} → {jitter_after:.5f}'
          f'（越小越平滑，窗口={SMOOTH_WIN}）')
    return df


def step4_outlier(df):
    """
    ④ 突变异常剔除：某关键点相邻帧位移突然超过 3 倍标准差，
       说明检测错乱，把该点该帧坐标置为缺失。
       必须在归一化之前做，否则异常会被归一化放大。
    """
    removed = 0
    for i in range(NUM_POINTS):
        dx = df[f'x{i}'].diff().abs()
        dy = df[f'y{i}'].diff().abs()
        dist = np.sqrt(dx ** 2 + dy ** 2)
        thr = dist.std() * Z_TH
        if np.isnan(thr) or thr == 0:
            continue
        bad = dist > thr
        df.loc[bad, [f'x{i}', f'y{i}', f'z{i}']] = np.nan
        removed += int(bad.sum())
    print(f'④ 突变异常剔除：共剔除 {removed} 个异常关键点坐标')
    return df


def step5_normalize(df):
    """
    ⑤ 归一化（最关键）：以骨盆(髋部中点)为原点、以髋宽为尺度，
       消除摄像头距离远近和人体身材差异对坐标数值的影响。
       关键点索引：23,24 为左右髋。
    """
    hip_cx = (df['x23'] + df['x24']) / 2
    hip_cy = (df['y23'] + df['y24']) / 2
    hip_cz = (df['z23'] + df['z24']) / 2
    hip_w = np.sqrt((df['x23'] - df['x24']) ** 2
                    + (df['y23'] - df['y24']) ** 2)
    hip_w = hip_w.replace(0, np.nan)   # 髋宽为0(侧身)时置空，避免除零

    for i in range(NUM_POINTS):
        df[f'x{i}'] = (df[f'x{i}'] - hip_cx) / hip_w
        df[f'y{i}'] = (df[f'y{i}'] - hip_cy) / hip_w
        df[f'z{i}'] = (df[f'z{i}'] - hip_cz) / hip_w

    # 归一化后再次清理极端值：坐标超出 ±10 视为异常(髋宽过小放大导致)，
    # 置为缺失，防止单个异常点毁掉整个特征
    coord_cols = [f'{c}{i}' for i in range(NUM_POINTS) for c in 'xyz']
    for c in coord_cols:
        df.loc[df[c].abs() > 10, c] = np.nan

    valid = df[['x23', 'y23']].dropna()
    print(f'⑤ 归一化：以骨盆为原点、髋宽为尺度，'
          f'坐标范围 {valid["x23"].min():.2f}~{valid["x23"].max():.2f}')
    return df


def step6_labels(df):
    """⑥ 标签统计：检查各类样本数量是否平衡。"""
    print('⑥ 标签分布：')
    if 'label' in df.columns:
        print('   ' + df['label'].value_counts().to_string().replace('\n', '\n   '))
    else:
        print('   未找到 label 列，请自行补充跌倒/正常标注。')
    return df


def main():
    print('开始清洗关键点数据...\n')
    df = load_data()
    print(f'原始数据：{len(df)} 帧，{df.shape[1]} 列\n')

    df = step1_missing(df)
    df = step2_drop_bad_frames(df)
    df = step3_smooth(df)
    df = step4_outlier(df)
    df = step5_normalize(df)
    df = step6_labels(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f'\n清洗完成！结果已保存到：{OUTPUT_FILE}')
    print(f'清洗后数据：{len(df)} 帧')


if __name__ == '__main__':
    main()
