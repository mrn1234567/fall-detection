# -*- coding: utf-8 -*-
"""复核清洗结果：读取清洗前后两个 CSV，检查关键指标是否合理。"""
import pandas as pd
import numpy as np

before = pd.read_csv('pose_data.csv')
after = pd.read_csv('cleaned_pose_data.csv')
coord_cols = [f'{c}{i}' for i in range(33) for c in 'xyz']

print('=' * 52)
print('数据清洗结果复核')
print('=' * 52)

print('\n【1. 帧数变化（低质量帧被剔除）】')
print(f'   清洗前 {len(before)} 帧 → 清洗后 {len(after)} 帧，'
      f'剔除 {len(before) - len(after)} 帧')

print('\n【2. 标签分布（各类样本数）】')
print('   清洗前：', dict(before['label'].value_counts()))
print('   清洗后：', dict(after['label'].value_counts()))

print('\n【3. 缺失值标记（低置信度坐标被置空）】')
nan_after = int(after[coord_cols].isna().sum().sum())
print(f'   清洗后标记缺失坐标 {nan_after} 个，这些在训练时会被忽略')

print('\n【4. 归一化有效性（骨盆原点、髋宽尺度）】')
# 用髋部(23)和肩部(11)做合理性检查，用分位数避免被单个极端值带偏
x23 = after['x23'].dropna()
y23 = after['y23'].dropna()
x11 = after['x11'].dropna()
print(f'   髋部 x 范围：{x23.min():.2f} ~ {x23.max():.2f}')
print(f'   髋部 y 范围：{y23.min():.2f} ~ {y23.max():.2f}')
print(f'   肩部 x 范围(1%~99%)：{x11.quantile(0.01):.2f} ~ {x11.quantile(0.99):.2f}')
print('   正常归一化后坐标应落在约 -3 ~ 3 之间（过大说明还有异常残留）')

print('\n【5. 清洗前后数据完整性】')
print(f'   清洗后仍有 {len(after)} 帧 / {after.shape[1]} 列，'
      f'可用于训练分类模型')

print('\n' + '=' * 52)
print('复核完成：若坐标范围合理、帧数减少、缺失已标记，'
      '说明清洗有效。')
print('=' * 52)
