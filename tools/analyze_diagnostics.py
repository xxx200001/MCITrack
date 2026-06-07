#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_diagnostics.py
======================
分析 tracker 写出的逐帧诊断 CSV (cfg.TEST.LOG_DIAGNOSTICS=True 生成于 DIAG_DIR)。

它回答一个核心问题:
    "我的 consistency 信号是 conf/APCE 的复读 (冗余), 还是能抓到它们抓不到的失败?"

用法:
    python tools/analyze_diagnostics.py --diag_dir ./diag_logs
    # 若有真值, 可做"失败检测"分析 (最有说服力):
    python tools/analyze_diagnostics.py --diag_dir ./diag_logs --gt_dir /path/to/lasot_gt
    # gt 每个序列一个文件: <gt_dir>/<seq>.txt 或 <gt_dir>/<seq>/groundtruth.txt
    # 每行 x,y,w,h (逗号/空格/制表符分隔), 第 0 行是初始帧

只依赖 numpy + 标准库; matplotlib 可选 (有则出图)。
"""

import argparse
import csv
import glob
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ----------------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------------
def load_csv(path):
    """读一个序列 CSV -> dict[col] = list(原始字符串)。"""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    header = rows[0]
    cols = {h: [] for h in header}
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        for h, v in zip(header, r):
            cols[h].append(v)
    return cols


def to_float(lst):
    out = []
    for v in lst:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def parse_v_per_template(lst):
    """';' 分隔 -> 每帧 (max, min, spread)。"""
    vmax, vmin, vspread = [], [], []
    for s in lst:
        try:
            vals = [float(x) for x in str(s).split(";") if x != ""]
        except ValueError:
            vals = []
        if vals:
            vmax.append(max(vals)); vmin.append(min(vals)); vspread.append(max(vals) - min(vals))
        else:
            vmax.append(np.nan); vmin.append(np.nan); vspread.append(np.nan)
    return np.array(vmax), np.array(vmin), np.array(vspread)


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------
# 真值 / IoU
# ----------------------------------------------------------------------------
def find_gt_file(gt_dir, seq):
    cands = [os.path.join(gt_dir, seq + ".txt"),
             os.path.join(gt_dir, seq, "groundtruth.txt"),
             os.path.join(gt_dir, seq, "groundtruth_rect.txt")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def load_gt(path):
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip().replace(",", " ").replace("\t", " ")
            if not line:
                continue
            parts = line.split()
            try:
                boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except (ValueError, IndexError):
                boxes.append([np.nan] * 4)
    return np.asarray(boxes, dtype=float)  # (N, 4) xywh, 第 0 行是初始帧


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    uni = aw * ah + bw * bh - inter
    return inter / uni if uni > 0 else 0.0


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_dir", required=True, help="逐帧 CSV 所在目录")
    ap.add_argument("--gt_dir", default=None, help="(可选) 真值目录, 开启失败检测分析")
    ap.add_argument("--out_dir", default=None, help="(可选) 图表/清单输出目录, 默认 diag_dir/analysis")
    ap.add_argument("--iou_fail", type=float, default=0.5, help="IoU 低于此值视为失败")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.diag_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.diag_dir, "*.csv")))
    if not files:
        print("没有找到 CSV:", args.diag_dir); return
    print("发现 %d 个序列 CSV" % len(files))

    # 聚合所有序列
    agg = {k: [] for k in ["conf", "cons", "cal", "apce", "vmax", "vmin", "vspread",
                            "mem_added", "h_final", "iou"]}
    disagreement = []   # conf 高但 consistency 低
    dirty_adds = []     # 低一致性时却把模板收进记忆池
    per_seq = []

    for path in files:
        seq = os.path.splitext(os.path.basename(path))[0]
        c = load_csv(path)
        if c is None:
            continue
        conf = to_float(c.get("conf_score", []))
        cons = to_float(c.get("consistency", []))
        cal = to_float(c.get("calibrated_consistency", []))
        apce = to_float(c.get("apce", [])) if "apce" in c else np.full_like(conf, np.nan)
        fid = to_float(c.get("frame_id", []))
        mem_added = to_float(c.get("memory_added", [])) if "memory_added" in c else np.zeros_like(conf)
        h_final = to_float(c.get("final_h_reset", [])) if "final_h_reset" in c else np.zeros_like(conf)
        vmax, vmin, vspread = parse_v_per_template(c.get("v_per_template", [""] * len(conf)))
        x = to_float(c.get("x", [])); y = to_float(c.get("y", []))
        w = to_float(c.get("w", [])); h = to_float(c.get("h", []))

        # IoU (可选)
        iou = np.full_like(conf, np.nan)
        if args.gt_dir:
            gp = find_gt_file(args.gt_dir, seq)
            if gp is not None:
                gt = load_gt(gp)
                for i in range(len(conf)):
                    k = int(fid[i]) if np.isfinite(fid[i]) else -1
                    if 0 <= k < len(gt) and np.all(np.isfinite(gt[k])):
                        iou[i] = iou_xywh([x[i], y[i], w[i], h[i]], gt[k])

        for key, arr in [("conf", conf), ("cons", cons), ("cal", cal), ("apce", apce),
                         ("vmax", vmax), ("vmin", vmin), ("vspread", vspread),
                         ("mem_added", mem_added), ("h_final", h_final), ("iou", iou)]:
            agg[key].append(arr)

        # 本序列内的分位阈值
        if np.isfinite(conf).sum() >= 5:
            conf_hi = np.nanpercentile(conf, 70)
            cons_lo = np.nanpercentile(cons, 30)
            for i in range(len(conf)):
                if conf[i] >= conf_hi and cons[i] <= cons_lo:
                    disagreement.append((seq, int(fid[i]), round(conf[i], 3), round(cons[i], 3),
                                         round(iou[i], 3) if np.isfinite(iou[i]) else ""))
            cons_q25 = np.nanpercentile(cons, 25)
            for i in range(len(conf)):
                if mem_added[i] >= 0.5 and cons[i] <= cons_q25:
                    dirty_adds.append((seq, int(fid[i]), round(conf[i], 3), round(cons[i], 3),
                                       round(iou[i], 3) if np.isfinite(iou[i]) else ""))

        per_seq.append((seq, len(conf), float(np.nanmean(conf)), float(np.nanmean(cons)),
                        int(np.nansum(h_final)),
                        float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")))

    A = {k: np.concatenate(v) for k, v in agg.items() if len(v)}

    # --- 报告 ---
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 70)
    emit("总帧数: %d   序列数: %d" % (len(A["conf"]), len(files)))
    emit("-" * 70)
    emit("各信号分布 (mean / std / min / max):")
    for name, key in [("conf", "conf"), ("consistency", "cons"),
                      ("calibrated_cons", "cal"), ("apce", "apce"), ("v_spread", "vspread")]:
        a = A[key]; a = a[np.isfinite(a)]
        if a.size:
            emit("  %-16s %.4f / %.4f / %.4f / %.4f" % (name, a.mean(), a.std(), a.min(), a.max()))

    emit("-" * 70)
    emit("相关性 (Pearson r;  > 0.85 视为高度冗余):")
    emit("  conf      vs consistency : %.3f   <-- 这条最关键" % pearson(A["conf"], A["cons"]))
    emit("  conf      vs calibrated  : %.3f" % pearson(A["conf"], A["cal"]))
    emit("  apce      vs consistency : %.3f" % pearson(A["apce"], A["cons"]))
    emit("  conf      vs apce        : %.3f" % pearson(A["conf"], A["apce"]))

    if args.gt_dir and np.isfinite(A["iou"]).any():
        iou = A["iou"]; conf = A["conf"]; cons = A["cons"]; apce = A["apce"]
        m = np.isfinite(iou)
        emit("-" * 70)
        emit("失败检测分析 (IoU<%.2f 记为失败; 有 %d 帧有真值):" % (args.iou_fail, m.sum()))
        emit("  corr(conf, IoU)        : %.3f" % pearson(conf, iou))
        emit("  corr(consistency, IoU) : %.3f   <-- 越高越说明一致性预示成败" % pearson(cons, iou))
        emit("  corr(apce, IoU)        : %.3f" % pearson(apce, iou))

        fail = m & (iou < args.iou_fail)
        nfail = int(fail.sum())
        if nfail > 0:
            conf_med = np.nanmedian(conf[m]); cons_med = np.nanmedian(cons[m])
            # conf 以为没事 (高于中位) 但一致性报警 (低于中位) 的失败帧 —— 这就是"补 CIF 的洞"
            caught = fail & (conf >= conf_med) & (cons <= cons_med)
            missed_both = fail & (conf >= conf_med) & (cons >= cons_med)
            emit("  失败帧数: %d" % nfail)
            emit("  其中 conf 高(>中位)的失败帧: %d" % int((fail & (conf >= conf_med)).sum()))
            emit("    └─ 被 consistency 抓到(低<中位): %d  (%.1f%%)  <-- 头条数字" %
                 (int(caught.sum()), 100.0 * caught.sum() / max(1, int((fail & (conf >= conf_med)).sum()))))
            emit("    └─ conf 和 consistency 都漏了 : %d" % int(missed_both.sum()))

    emit("-" * 70)
    emit("conf 高 / consistency 低 的「打架」帧: %d  (详见 disagreement_frames.csv)" % len(disagreement))
    emit("低一致性却被收进记忆池的「脏模板」帧: %d  (详见 dirty_template_adds.csv)" % len(dirty_adds))
    emit("=" * 70)

    # --- 落盘 ---
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _dump(os.path.join(out_dir, "disagreement_frames.csv"),
          ["seq", "frame_id", "conf", "consistency", "iou"], disagreement)
    _dump(os.path.join(out_dir, "dirty_template_adds.csv"),
          ["seq", "frame_id", "conf", "consistency", "iou"], dirty_adds)
    _dump(os.path.join(out_dir, "per_sequence.csv"),
          ["seq", "frames", "mean_conf", "mean_cons", "n_h_reset", "mean_iou"], per_seq)

    # --- 图 ---
    if HAS_PLT:
        _scatter(A, out_dir, args.gt_dir is not None)
        print("图已保存到:", out_dir)
    else:
        print("(未安装 matplotlib, 跳过画图)")
    print("分析结果目录:", out_dir)


def _dump(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _scatter(A, out_dir, has_iou):
    conf, cons = A["conf"], A["cons"]
    m = np.isfinite(conf) & np.isfinite(cons)
    plt.figure(figsize=(6, 5))
    if has_iou and np.isfinite(A["iou"]).any():
        iou = A["iou"]
        mm = m & np.isfinite(iou)
        sc = plt.scatter(conf[mm], cons[mm], c=iou[mm], cmap="RdYlGn", s=6, vmin=0, vmax=1)
        plt.colorbar(sc, label="IoU")
    else:
        plt.scatter(conf[m], cons[m], s=6, alpha=0.4)
    plt.xlabel("conf_score"); plt.ylabel("consistency")
    plt.title("conf vs consistency (绿=跟对 红=跟丢)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_conf_vs_consistency.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
