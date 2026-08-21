# Preliminary Legacy Data Inventory

> Generated automatically. Units, frames, and semantics remain unverified.

- Source root: `/public/home/v-chengwy/cjz/RL_credit-assign/Mechanism-Guided-Residual-Reinforcement-Learning-for-Robotic-Hemming/data/DATA/DATA`
- CSV files: 63
- Rows: 8912
- Exact duplicate hash groups: 0

## Robot summary

| Robot | Files | Rows |
|---|---:|---:|
| kuka_kr210_r2700_2 | 30 | 5240 |
| kuka_kr210_r3100_2 | 6 | 460 |
| kuka_kr240_r3330 | 9 | 775 |
| unverified | 2 | 277 |
| ur5 | 16 | 2160 |

## Files requiring review

- `KR210 R2700-2/20250522/CSV/data_all+手摇10个.csv`: merged_file_candidate_do_not_combine_with_components
- `KR210 R2700-2/20250522/CSV/data_all.csv`: merged_file_candidate_do_not_combine_with_components
- `KR210 R2700-2/20250529/CSV/data_all.csv`: merged_file_candidate_do_not_combine_with_components
- `KUKA_KR240R3330/Hemm1.csv`: missing_or_nonstandard_real_xyz_headers
- `KUKA_KR240R3330/建模数据1.csv`: real_column_order_is_['b-real', 'c-real', '']
- `KUKA_KR240R3330/建模数据2.csv`: real_column_order_is_['b-real', 'c-real', '']
- `UR5_DATA/20250714/建模数据.csv`: real_column_order_is_['y-real', 'z-real', '']
- `UR5_DATA/20250807/10Pos.csv`: real_column_order_is_['rx-real', 'ry-real', 'rz-real']
- `UR5_DATA/20250807/建模数据.csv`: real_column_order_is_['rx-real', 'ry-real', 'rz-real']
- `UR5_DATA/20250807/验证数据(1).csv`: real_column_order_is_['rx-real', 'ry-real', 'rz-real']
- `UR5_DATA/data08.csv`: real_column_order_is_['z-real', 'y-real', 'x-real']; declared_header_mapping_much_worse_than_positional_xyz_mapping
- `UR5_DATA/data_all.csv`: merged_file_candidate_do_not_combine_with_components
- `test00.csv`: unknown_robot_frame_and_abnormal_scale_requires_confirmation
- `train00.csv`: unknown_robot_frame_and_abnormal_scale_requires_confirmation

## File metrics

| File | Robot | Rows | Declared median | Positional median |
|---|---|---:|---:|---:|
| `KR210 R2700-2/20250522/CSV/data01.csv` | kuka_kr210_r2700_2 | 100 | 5.8079 | 5.8079 |
| `KR210 R2700-2/20250522/CSV/data010.csv` | kuka_kr210_r2700_2 | 100 | 7.3848 | 7.3848 |
| `KR210 R2700-2/20250522/CSV/data011.csv` | kuka_kr210_r2700_2 | 100 | 7.0101 | 7.0101 |
| `KR210 R2700-2/20250522/CSV/data02.csv` | kuka_kr210_r2700_2 | 100 | 6.5553 | 6.5553 |
| `KR210 R2700-2/20250522/CSV/data03.csv` | kuka_kr210_r2700_2 | 100 | 6.7313 | 6.7313 |
| `KR210 R2700-2/20250522/CSV/data04.csv` | kuka_kr210_r2700_2 | 100 | 6.7166 | 6.7166 |
| `KR210 R2700-2/20250522/CSV/data05.csv` | kuka_kr210_r2700_2 | 100 | 6.4587 | 6.4587 |
| `KR210 R2700-2/20250522/CSV/data06.csv` | kuka_kr210_r2700_2 | 100 | 6.9973 | 6.9973 |
| `KR210 R2700-2/20250522/CSV/data07.csv` | kuka_kr210_r2700_2 | 100 | 6.8467 | 6.8467 |
| `KR210 R2700-2/20250522/CSV/data08.csv` | kuka_kr210_r2700_2 | 100 | 7.1641 | 7.1641 |
| `KR210 R2700-2/20250522/CSV/data09.csv` | kuka_kr210_r2700_2 | 100 | 6.8373 | 6.8373 |
| `KR210 R2700-2/20250522/CSV/data_all+手摇10个.csv` | kuka_kr210_r2700_2 | 1010 | 6.7991 | 6.7991 |
| `KR210 R2700-2/20250522/CSV/data_all.csv` | kuka_kr210_r2700_2 | 1000 | 6.7381 | 6.7381 |
| `KR210 R2700-2/20250522/CSV/test.csv` | kuka_kr210_r2700_2 | 3 | 6.5250 | 6.5250 |
| `KR210 R2700-2/20250522/CSV/手摇10个.csv` | kuka_kr210_r2700_2 | 10 | 9.5887 | 9.5887 |
| `KR210 R2700-2/20250529/CSV/Newdata011.csv` | kuka_kr210_r2700_2 | 97 | 6.5613 | 6.5613 |
| `KR210 R2700-2/20250529/CSV/data01.csv` | kuka_kr210_r2700_2 | 100 | 6.4429 | 6.4429 |
| `KR210 R2700-2/20250529/CSV/data010.csv` | kuka_kr210_r2700_2 | 98 | 6.7278 | 6.7278 |
| `KR210 R2700-2/20250529/CSV/data02.csv` | kuka_kr210_r2700_2 | 100 | 6.9222 | 6.9222 |
| `KR210 R2700-2/20250529/CSV/data03.csv` | kuka_kr210_r2700_2 | 98 | 5.8563 | 5.8563 |
| `KR210 R2700-2/20250529/CSV/data04.csv` | kuka_kr210_r2700_2 | 100 | 7.5814 | 7.5814 |
| `KR210 R2700-2/20250529/CSV/data05.csv` | kuka_kr210_r2700_2 | 100 | 5.7296 | 5.7296 |
| `KR210 R2700-2/20250529/CSV/data06.csv` | kuka_kr210_r2700_2 | 100 | 6.6380 | 6.6380 |
| `KR210 R2700-2/20250529/CSV/data07.csv` | kuka_kr210_r2700_2 | 98 | 6.7778 | 6.7778 |
| `KR210 R2700-2/20250529/CSV/data08.csv` | kuka_kr210_r2700_2 | 100 | 5.9955 | 5.9955 |
| `KR210 R2700-2/20250529/CSV/data09.csv` | kuka_kr210_r2700_2 | 100 | 7.7600 | 7.7600 |
| `KR210 R2700-2/20250529/CSV/data_all.csv` | kuka_kr210_r2700_2 | 1004 | 6.6421 | 6.6421 |
| `KR210 R2700-2/20250529/CSV/test.csv` | kuka_kr210_r2700_2 | 2 | 7.9480 | 7.9480 |
| `KR210 R2700-2/20250529/CSV/手摇10个.csv` | kuka_kr210_r2700_2 | 10 | 0.8089 | 0.8089 |
| `KR210 R2700-2/20250529/CSV/手摇10验证.csv` | kuka_kr210_r2700_2 | 10 | 1.4588 | 1.4588 |
| `KR210-R3100-2-FLR_8163181_20250608/CSV/建模数据1.csv` | kuka_kr210_r3100_2 | 62 | 0.8674 | 0.8674 |
| `KR210-R3100-2-FLR_8163181_20250608/CSV/建模数据2.csv` | kuka_kr210_r3100_2 | 68 | 0.7336 | 0.7336 |
| `KR210-R3100-2-FLR_8163181_20250608/CSV/建模数据3.csv` | kuka_kr210_r3100_2 | 59 | 0.8331 | 0.8331 |
| `KR210-R3100-2-FLR_8163181_20250608/CSV/建模数据4.csv` | kuka_kr210_r3100_2 | 72 | 0.8471 | 0.8471 |
| `KR210-R3100-2-FLR_8163181_20250608/CSV/建模数据_x1x2x3.csv` | kuka_kr210_r3100_2 | 189 | 0.8331 | 0.8331 |
| `KR210-R3100-2-FLR_8163181_20250608/CSV/建模数据潘.csv` | kuka_kr210_r3100_2 | 10 | 0.9692 | 0.9692 |
| `KUKA_KR240R3330/Hemm1.csv` | kuka_kr240_r3330 | 92 |  | 2637.4870 |
| `KUKA_KR240R3330/test.csv` | kuka_kr240_r3330 | 2 | 2.5804 | 2.5804 |
| `KUKA_KR240R3330/建模数据1.csv` | kuka_kr240_r3330 | 48 | 2.2998 | 2520.6362 |
| `KUKA_KR240R3330/建模数据2.csv` | kuka_kr240_r3330 | 74 | 1.9044 | 2148.8219 |
| `KUKA_KR240R3330/建模数据3.csv` | kuka_kr240_r3330 | 70 | 1.8580 | 1.8580 |
| `KUKA_KR240R3330/建模数据4.csv` | kuka_kr240_r3330 | 81 | 1.8016 | 1.8016 |
| `KUKA_KR240R3330/建模数据5.csv` | kuka_kr240_r3330 | 68 | 1.7188 | 1.7188 |
| `KUKA_KR240R3330/建模数据_x1x2x3x4.csv` | kuka_kr240_r3330 | 273 | 1.8813 | 1.8813 |
| `KUKA_KR240R3330/验证数据.csv` | kuka_kr240_r3330 | 67 | 1.7136 | 1.7136 |
| `UR5_DATA/20250714/建模数据.csv` | ur5 | 100 | 1.9978 |  |
| `UR5_DATA/20250806/10.csv` | ur5 | 10 | 2.6060 | 2.6060 |
| `UR5_DATA/20250806/建模数据1.csv` | ur5 | 115 | 1.3323 | 1.3323 |
| `UR5_DATA/20250806/验证数据.csv` | ur5 | 28 | 1.5223 | 1.5223 |
| `UR5_DATA/20250807/10Pos.csv` | ur5 | 9 | 2.6386 | 690.2015 |
| `UR5_DATA/20250807/建模数据.csv` | ur5 | 351 | 4.0127 | 462.0866 |
| `UR5_DATA/20250807/验证数据(1).csv` | ur5 | 96 | 3.8558 | 462.7515 |
| `UR5_DATA/data01.csv` | ur5 | 91 | 2.0829 | 2.0829 |
| `UR5_DATA/data02.csv` | ur5 | 92 | 2.4230 | 2.4230 |
| `UR5_DATA/data03.csv` | ur5 | 90 | 2.2692 | 2.2692 |
| `UR5_DATA/data04.csv` | ur5 | 89 | 2.1413 | 2.1413 |
| `UR5_DATA/data05.csv` | ur5 | 91 | 2.2297 | 2.2297 |
| `UR5_DATA/data06.csv` | ur5 | 90 | 2.1739 | 2.1739 |
| `UR5_DATA/data07.csv` | ur5 | 91 | 2.3343 | 2.3343 |
| `UR5_DATA/data08.csv` | ur5 | 93 | 1751.7064 | 2.0606 |
| `UR5_DATA/data_all.csv` | ur5 | 724 | 2.2109 | 2.2109 |
| `test00.csv` | unverified | 81 | 1.7501 | 1.7501 |
| `train00.csv` | unverified | 196 | 1.8529 | 1.8529 |
