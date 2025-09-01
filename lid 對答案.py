import os

# ================================
PRED_DAPU = "輸出/result/大埔腔.txt"
PRED_ZHAOAN = "輸出/result/詔安腔.txt"

GT_DAPU_DIR = "驗證/大埔腔"
GT_ZHAOAN_DIR = "驗證/詔安腔"

OUT_DIR = "對答案"
# ================================

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_pred_list(path, label_name):
    preds = {}
    if not os.path.exists(path):
        print(f"[警告] 預測檔不存在: {path}")
        return preds
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            base = os.path.splitext(os.path.basename(ln))[0]
            if base not in preds:
                preds[base] = set()
            preds[base].add(label_name)
    return preds

def build_gt_index(dir_path, label_name):
    index = {}
    if not os.path.exists(dir_path):
        print(f"[警告] GT 目錄不存在: {dir_path}")
        return index
    for root, _, files in os.walk(dir_path):
        for fn in files:
            base = os.path.splitext(os.path.basename(fn))[0]
            full = os.path.join(root, fn)
            if base not in index:
                index[base] = []
            index[base].append((label_name, full))
    return index

def main():
    ensure_dir(OUT_DIR)

    preds_dapu = read_pred_list(PRED_DAPU, "大埔腔")
    preds_zhaoan = read_pred_list(PRED_ZHAON if False else PRED_ZHAOAN, "詔安腔")  # safe reference

    preds = {}
    for d in (preds_dapu, preds_zhaoan):
        for k, s in d.items():
            if k not in preds:
                preds[k] = set()
            preds[k].update(s)

    total_predictions = len(preds)

    gt_dapu = build_gt_index(GT_DAPU_DIR, "大埔腔")
    gt_zhaoan = build_gt_index(GT_ZHAOAN_DIR, "詔安腔")

    gt_index = {}
    for d in (gt_dapu, gt_zhaoan):
        for k, v in d.items():
            if k not in gt_index:
                gt_index[k] = []
            gt_index[k].extend(v)

    ambiguous = []
    for k, entries in gt_index.items():
        labels = set(lbl for lbl, _ in entries)
        if len(labels) > 1:
            ambiguous.append((k, entries))

    # 評估
    matched = 0
    correct = 0
    errors = []
    missing_gt = []
    confusion = {}

    for base, pred_labels in preds.items():
        if base not in gt_index:
            missing_gt.append(base)
            continue

        matched += 1

        gt_entries = gt_index[base]
        gt_labels = set(lbl for lbl, _ in gt_entries)
        is_correct = any(pl in gt_labels for pl in pred_labels)

        for pl in pred_labels:
            for true_lbl, _ in gt_entries:
                key = (pl, true_lbl)
                confusion[key] = confusion.get(key, 0) + 1

        if is_correct:
            correct += 1
        else:
            gt_paths = [p for _, p in gt_entries]
            errors.append((base, sorted(list(pred_labels)), sorted(list(gt_labels)), gt_paths))

    strict_accuracy = correct / total_predictions if total_predictions > 0 else 0.0
    matched_accuracy = correct / matched if matched > 0 else 0.0

    err_path = os.path.join(OUT_DIR, "error.txt")
    with open(err_path, "w", encoding="utf-8") as ef:
        ef.write("錯誤分類清單（預測標籤 != 真實標籤）\n")
        ef.write("格式: 檔名 \t 預測標籤(可能多個) \t 真實標籤(可能多個) \t 真實檔案路徑(s)\n\n")
        for base, pred_labels, gt_labels, gt_paths in errors:
            ef.write(f"{base}\t{','.join(pred_labels)}\t{','.join(gt_labels)}\t{';'.join(gt_paths)}\n")

    missing_path = os.path.join(OUT_DIR, "missing_groundtruth.txt")
    with open(missing_path, "w", encoding="utf-8") as mf:
        mf.write("預測清單中未在任何 ground-truth 目錄找到的檔名（未配對成功）\n")
        mf.write("每行一個 basename（不含副檔名）\n\n")
        for b in missing_gt:
            mf.write(b + "\n")

    amb_path = os.path.join(OUT_DIR, "ambiguous_groundtruth.txt")
    with open(amb_path, "w", encoding="utf-8") as af:
        af.write("同一 basename 出現在多個 ground-truth 資料夾（需人工檢查）\n")
        af.write("格式: 檔名 \t (標籤|路徑);(標籤|路徑);...\n\n")
        for base, entries in ambiguous:
            pairs = ["%s|%s" % (lbl, path) for lbl, path in entries]
            af.write(f"{base}\t{';'.join(pairs)}\n")

    summary_path = os.path.join(OUT_DIR, "accuracy_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("準確率摘要\n")
        sf.write("========\n")
        sf.write(f"總預測數（預測清單中獨立 basename）: {total_predictions}\n")
        sf.write(f"已配對到 ground-truth（matched）: {matched}\n")
        sf.write(f"未配對到 ground-truth（missing）: {len(missing_gt)} （見 {os.path.basename(missing_path)}）\n")
        sf.write(f"判定正確數: {correct}\n")
        sf.write(f"嚴格準確率 (正確 / 總預測數): {strict_accuracy:.4f}\n")
        sf.write(f"已配對準確率 (正確 / 已配對數): {matched_accuracy:.4f}\n\n")

        sf.write("混淆統計（預測 -> 真實）:\n")
        # sort keys for deterministic output
        for (p, t), cnt in sorted(confusion.items(), key=lambda x:(x[0][0], x[0][1])):
            sf.write(f"  {p} -> {t}: {cnt}\n")

        sf.write("\n檔案統計:\n")
        sf.write(f"  錯誤分類數量 (error.txt): {len(errors)}\n")
        sf.write(f"  未配對數量 (missing_groundtruth.txt): {len(missing_gt)}\n")
        sf.write(f"  ambiguous (ambiguous_groundtruth.txt): {len(ambiguous)}\n")
        sf.write("\n已寫入檔案：\n")
        sf.write(f"  {os.path.basename(err_path)}\n")
        sf.write(f"  {os.path.basename(missing_path)}\n")
        sf.write(f"  {os.path.basename(amb_path)}\n")
        sf.write(f"  {os.path.basename(summary_path)}\n")

    # console（中文）輸出
    print("=== 準確率報告 ===")
    print(f"總預測數: {total_predictions}")
    print(f"已配對到 GT: {matched}")
    print(f"未配對到 GT: {len(missing_gt)} （見 {missing_path}）")
    print(f"判定正確數: {correct}")
    print(f"嚴格準確率: {strict_accuracy:.4f}")
    print(f"已配對準確率: {matched_accuracy:.4f}")
    print(f"錯誤分類數量: {len(errors)} （見 {err_path}）")
    print(f"ambiguous ground-truth 條目數: {len(ambiguous)} （見 {amb_path}）")
    print(f"摘要已寫入: {summary_path}")

if __name__ == "__main__":
    main()
