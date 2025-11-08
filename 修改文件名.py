import os
import re

base_path = "/media/sdb/time_series/TSF_proj/checkpoints"


import os
from typing import List

def list_dirs(base_path: str, keyword: str, recursive: bool = False, full_path: bool = False) -> List[str]:
    """
    列出 base_path 下的文件夹名称（或完整路径）。
    :param base_path: 要遍历的根目录
    :param keyword:   可选，只有名称包含该关键字的文件夹才会被返回（大小写敏感）
    :param recursive: 是否递归遍历子目录
    :param full_path: 返回值是否使用绝对/完整路径（默认返回名称）
    :return:          符合条件的文件夹列表（按字母序）
    """
    results = []

    if recursive:
        for root, dirs, _ in os.walk(base_path):
            for d in dirs:
                if keyword and keyword not in d:
                    continue
                results.append(os.path.join(root, d) if full_path else d)
    else:
        for d in os.listdir(base_path):
            p = os.path.join(base_path, d)
            if os.path.isdir(p) and (keyword is None or keyword in d):
                results.append(p if full_path else d)

    return sorted(results)

# 只收集包含 iTransformer 的文件夹
itransformer_dirs = list_dirs(base_path, keyword="iTransformer")

print(f"找到 {len(itransformer_dirs)} 个包含 iTransformer 的文件夹：")
for d in itransformer_dirs:
    print("  -", d)

# 关键：第一个 iTransformer 属于 model_id
# model_id = 从开头到第一个 'iTransformer'（包含它）
# 后面再提取 ft/sl/ll/pl
pattern = re.compile(
    r'^(?P<model_id>.+?iTransformer)_'             # 到第一个 iTransformer 为止（含）
    r'(?P<model>[^_]+)_'                             # model，例如 iTransformer
    r'(?P<data>.+?)_'                              # data，例如 solar_data
    r'bs(?P<batch_size>\d+)_'                      # bs16 → batch_size=16
    r'ft(?P<features>\w+)_'                        # ftMS → features=MS
    r'sl(?P<seq_len>\d+)_'                         # sl96 → seq_len=96
    r'll(?P<label_len>\d+)_'                       # ll48 → label_len=48
    r'pl(?P<pred_len>\d+)$'                        # pl24 → pred_len=24
)

print("\n✅ 提取结果（先检查）：")
for name in itransformer_dirs:
    m = pattern.match(name)
    if not m:
        print(f"⚠ 未匹配：{name}")
        continue

    g = m.groupdict()
    # 打印整个匹配结果字典
    print("提取字段:", g)

    # 你要的新 model_id 预览（不改名，只打印）
    new_model_id = f"solar_data"

    # 你的目标：setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}'.format(args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len)
    # 其中 model_id 采用规范化：solar_{seq_len}_{pred_len}
    setting_name = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}".format(
        new_model_id,
        g['model'],                                # args.model
        g['data'],                                 # args.data
        g['features'],                             # args.features
        g['seq_len'],                              # args.seq_len
        g['label_len'],                            # args.label_len
        g['pred_len']                              # args.pred_len
    )
    print(f"[PREVIEW] 将重命名为: {setting_name}\n")
    old_path = os.path.join(base_path, name)
    new_path = os.path.join(base_path, setting_name)
    if old_path != new_path:
        print(f"🔄 重命名: {old_path} → {new_path}")
        os.rename(old_path, new_path)



# ====== TimeBridge 目录收集与字段提取（不重命名，仅打印） ======
# 只收集包含 TimeBridge 的文件夹

timebridge_dirs = list_dirs(base_path, keyword="TimeBridge")

print("\n===== TimeBridge 目录提取预览 =====")
print(f"找到 {len(timebridge_dirs)} 个包含 TimeBridge 的文件夹：")
for d in timebridge_dirs:
    print("  -", d)

# TimeBridge 命名风格字段提取正则
pattern_tb = re.compile(
    r'^(?P<model_id>.+?_\d+_\d+)_'            # model_id：到 "_数字_数字" 为止
    r'(?P<model>[^_]+)_'                        # model：不含下划线（如 TimeBridge）
    r'(?P<data>.+?)_bs'                         # data：可含下划线，直到 _bs
    r'(?P<batch_size>\d+)_'                    # bs64 → 64
    r'ft(?P<features>[^_]+)_'                   # ftMS → MS
    r'sl(?P<seq_len>\d+)_'                     # sl96 → 96
    r'll(?P<label_len>\d+)_'                   # ll48 → 48
    r'pl(?P<pred_len>\d+)$'                    # pl1  → 1
)

print("\n✅ TimeBridge 提取结果（先检查）：")
for name in timebridge_dirs:
    m = pattern_tb.match(name)
    if not m:
        print(f"⚠ 未匹配：{name}")
        continue
    g = m.groupdict()
    print("提取字段:", g)

    new_model_id = f"solar_data"

    # 你的目标：setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}'.format(args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len)
    # 其中 model_id 采用规范化：solar_{seq_len}_{pred_len}
    setting_name = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}".format(
        new_model_id,
        g['model'],  # args.model
        g['data'],  # args.data
        g['features'],  # args.features
        g['seq_len'],  # args.seq_len
        g['label_len'],  # args.label_len
        g['pred_len']  # args.pred_len
    )
    print(f"[PREVIEW] 将重命名为: {setting_name}\n")
    old_path = os.path.join(base_path, name)
    new_path = os.path.join(base_path, setting_name)
    if old_path != new_path:
        print(f"🔄 重命名: {old_path} → {new_path}")
        os.rename(old_path, new_path)


cycleNet_dirs = list_dirs(base_path, keyword="CycleNet")
print("\n===== CycleNet 目录提取预览 =====")
print(f"找到 {len(cycleNet_dirs)} 个包含 CycleNet 的文件夹：")
for d in cycleNet_dirs:
    print("  -", d)
# CycleNet 命名风格字段提取正则
pattern_cy = re.compile(
    r'^(?P<model_id>[^_]+)_'             # model_id：test1
    r'(?P<model>[^_]+)_'                 # model：CycleNet
    r'(?P<data>.+?)_'                    # data：solar_data
    r'bs(?P<batch_size>\d+)_'            # bs16
    r'ft(?P<features>\w+)_'              # ftMS
    r'sl(?P<seq_len>\d+)_'               # sl96
    r'll(?P<label_len>\d+)_'             # ll48
    r'pl(?P<pred_len>\d+)$'              # pl1
)
print("\n✅ CycleNet 提取结果（先检查）：")
for name in cycleNet_dirs:
    m = pattern_cy.match(name)
    if not m:
        print(f"⚠ 未匹配：{name}")
        continue
    g = m.groupdict()
    print("提取字段:", g)
    new_model_id = f"solar_data"
   # 你的目标：setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}'.format(args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len)
    # 其中 model_id 采用规范化：solar_{seq_len}_{pred_len}
    setting_name = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}".format(
        new_model_id,
        g['model'],  # args.model
        g['data'],  # args.data
        g['features'],  # args.features
        g['seq_len'],  # args.seq_len
        g['label_len'],  # args.label_len
        g['pred_len']  # args.pred_len
    )
    print(f"[PREVIEW] 将重命名为: {setting_name}\n")
    old_path = os.path.join(base_path, name)
    new_path = os.path.join(base_path, setting_name)
    if old_path != new_path:
        print(f"🔄 重命名: {old_path} → {new_path}")
        os.rename(old_path, new_path)