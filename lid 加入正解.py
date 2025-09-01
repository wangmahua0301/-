import os
import csv
import re
import shutil
import evaluate
cer_metric = evaluate.load("wer")

# ================================================
MERGED_CSV = "data/合併.csv"
DAPU_CSV = "data/大埔腔.csv"
ZHAOAN_CSV = "data/詔安腔.csv"
ALL_CSV = "data/ALL.csv"

ROOT_WAV_DIR = "驗證"
OUTPUT_BASE_DIR = "輸出"
# ================================================

PUNCTUATION = (
    r"""[\s\.,，。！？、；:：\(\)（）\[\]【】「」『』‹›《》〈〉—\-…·・'\"“”‘’—―]"""
)

def normalize_text(s):
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(PUNCTUATION, "", s)
    return s

def read_two_col_csv(path):
    d = {}
    if not os.path.exists(path):
        print(f"[WARN] CSV not found: {path}")
        return d
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row:
                continue
            raw_key = row[0].strip()
            if raw_key == "":
                continue

            if first:
                first = False
                low0 = raw_key.lower()
                header_keywords = ("錄音檔檔名", "辨認出之客語漢字", "filename", "text", "檔名", "辨認")
                if any(k in low0 for k in header_keywords) or (len(row) >= 2 and any(k in (row[1] or "").lower() for k in header_keywords)):
                    continue

            key = os.path.splitext(os.path.basename(raw_key))[0]
            if len(row) >= 2:
                val = ",".join(row[1:]).strip()
            else:
                val = ""
            d[key] = val
    return d

def build_wav_index(root_dir):
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            base, ext = os.path.splitext(fn)
            if base not in index:
                index[base] = os.path.join(dirpath, fn)
    return index

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    ensure_dir(OUTPUT_BASE_DIR)
    dapu_out_dir = os.path.join(OUTPUT_BASE_DIR, "大埔腔")
    zhaoan_out_dir = os.path.join(OUTPUT_BASE_DIR, "詔安腔")
    mix_out_dir = os.path.join(OUTPUT_BASE_DIR, "合併")
    result_dir = os.path.join(OUTPUT_BASE_DIR, "result")
    ensure_dir(dapu_out_dir)
    ensure_dir(zhaoan_out_dir)
    ensure_dir(mix_out_dir)
    ensure_dir(result_dir)

    OUT_DAPU_LIST = os.path.join(result_dir, "大埔腔.txt")
    OUT_ZHAOAN_LIST = os.path.join(result_dir, "詔安腔.txt")
    OUT_MIX_LIST = os.path.join(result_dir, "合併.txt")
    OUT_SUMMARY = os.path.join(result_dir, "summary_all.txt")
    OUT_TIE = os.path.join(result_dir, "平手.txt")
    OUT_MISSING = os.path.join(result_dir, "missing_wav.txt")

    print("讀取 CSV ...")
    merged = read_two_col_csv(MERGED_CSV)
    dapu = read_two_col_csv(DAPU_CSV)
    zhaoan = read_two_col_csv(ZHAOAN_CSV)
    all_truth = read_two_col_csv(ALL_CSV)

    all_keys = set(merged.keys())
    all_keys.update(dapu.keys())
    all_keys.update(zhaoan.keys())

    print(f"總共發現 {len(all_keys)} 個檔案要判定。")

    wav_index = build_wav_index(ROOT_WAV_DIR)
    print(f"找到 {len(wav_index)} 個可對應的檔案。")

    out_dapu_list = []
    out_zhaoan_list = []
    tie_list = []
    tie_blocks_diff = []
    tie_blocks_same = []
    missing_wav = []
    lines_summary = []

    same_text_count = 0
    tie_true_cer_merged_zero = 0
    tie_true_cer_zhaoan_zero = 0
    tie_true_cer_dapu_zero = 0
    tie_all_three_zero = 0
    tie_all_three_equal = 0

    for key in sorted(all_keys):
        ref_raw = merged.get(key, "")
        dapu_raw = dapu.get(key, "")
        zhaoan_raw = zhaoan.get(key, "")
        true_ref_raw = all_truth.get(key, ref_raw)

        ref = normalize_text(ref_raw)
        dapu_n = normalize_text(dapu_raw)
        zhaoan_n = normalize_text(zhaoan_raw)
        true_ref = normalize_text(true_ref_raw)

        if dapu_n != "" and dapu_n == zhaoan_n:
            same_text_count += 1

        # === 改用 evaluate 計算 CER ===
        cer_dapu = 100 * cer_metric.compute(predictions=[dapu_n], references=[ref])
        cer_zhaoan = 100 * cer_metric.compute(predictions=[zhaoan_n], references=[ref])
        true_cer_dapu = 100 * cer_metric.compute(predictions=[dapu_n], references=[true_ref])
        true_cer_zhaoan = 100 * cer_metric.compute(predictions=[zhaoan_n], references=[true_ref])
        true_cer_merged = 100 * cer_metric.compute(predictions=[ref], references=[true_ref])

        # 判斷勝負
        if cer_dapu < cer_zhaoan:
            decision = "大埔腔"
            out_dapu_list.append(key)
            assigned_dir = dapu_out_dir
        elif cer_zhaoan < cer_dapu:
            decision = "詔安腔"
            out_zhaoan_list.append(key)
            assigned_dir = zhaoan_out_dir
        else:
            decision = "平手"
            tie_list.append(key)
            assigned_dir = mix_out_dir

            if true_cer_merged == 0:
                tie_true_cer_merged_zero += 1
            if true_cer_zhaoan == 0:
                tie_true_cer_zhaoan_zero += 1
            if true_cer_dapu == 0:
                tie_true_cer_dapu_zero += 1
            if true_cer_merged == 0 and true_cer_zhaoan == 0 and true_cer_dapu == 0:
                tie_all_three_zero += 1
            if dapu_n == zhaoan_n == ref != "":
                tie_all_three_equal += 1

        wav_path = wav_index.get(key)
        wav_copied_path = ""
        if wav_path:
            try:
                dest_path = os.path.join(assigned_dir, os.path.basename(wav_path))
                shutil.copy2(wav_path, dest_path)
                wav_copied_path = dest_path
            except Exception as e:
                print(f"[ERROR] Copy failed for {key}: {e}")
        else:
            missing_wav.append(key)

        block_lines = []
        block_lines.append(f"檔案名稱:{key}")
        block_lines.append(f"正解: {true_ref_raw}")
        block_lines.append(f"合併: {ref_raw}")
        block_lines.append(f"真正cer(合併): {true_cer_merged:.4f}")
        block_lines.append(f"詔安腔: {zhaoan_raw}")
        block_lines.append(f"cer: {cer_zhaoan:.4f}")
        block_lines.append(f"真正cer: {true_cer_zhaoan:.4f}")
        block_lines.append(f"大埔腔: {dapu_raw}")
        block_lines.append(f"cer: {cer_dapu:.4f}")
        block_lines.append(f"真正cer: {true_cer_dapu:.4f}")
        block_lines.append(f"判定: {decision}")
        if wav_copied_path:
            block_lines.append(f"複製至: {wav_copied_path}")
        else:
            block_lines.append("音檔狀態: 找不到對應 wav（未複製）")
        block = "\n".join(block_lines)
        lines_summary.append(block)

        if decision == "平手":
            if dapu_n != "" and dapu_n == zhaoan_n:
                tie_blocks_same.append(block)
            else:
                tie_blocks_diff.append(block)

    with open(OUT_DAPU_LIST, "w", encoding="utf-8") as f:
        for k in out_dapu_list:
            f.write(k + "\n")
    with open(OUT_ZHAOAN_LIST, "w", encoding="utf-8") as f:
        for k in out_zhaoan_list:
            f.write(k + "\n")
    with open(OUT_MIX_LIST, "w", encoding="utf-8") as f:
        for k in tie_list:
            f.write(k + "\n")

    tie_count = len(tie_list)
    with open(OUT_TIE, "w", encoding="utf-8") as f:
        f.write(f"兩者一樣的數量: {same_text_count} / 平手數量: {tie_count}\n")
        f.write(f"真正cer(合併)為0的數量: {tie_true_cer_merged_zero}\n")
        f.write(f"詔安腔真正cer為0的數量: {tie_true_cer_zhaoan_zero}\n")
        f.write(f"大埔腔真正cer為0的數量: {tie_true_cer_dapu_zero}\n")
        f.write(f"三者真正cer皆為0的數量: {tie_all_three_zero}\n")
        f.write(f"三者真正cer皆一樣的數量: {tie_all_three_equal}\n\n")

        f.write("=== 長不一樣平手 ===\n\n")
        if tie_blocks_diff:
            f.write("\n\n".join(tie_blocks_diff))
        else:
            f.write("(無)\n")
        f.write("\n\n=== 長一樣平手 ===\n\n")
        if tie_blocks_same:
            f.write("\n\n".join(tie_blocks_same))
        else:
            f.write("(無)\n")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines_summary))

    if missing_wav:
        with open(OUT_MISSING, "w", encoding="utf-8") as f:
            for k in missing_wav:
                f.write(k + "\n")

    print("完成")
    print(f" - {OUT_DAPU_LIST}")
    print(f" - {OUT_ZHAOAN_LIST}")
    print(f" - {OUT_MIX_LIST}")
    print(f" - {OUT_TIE}")
    if missing_wav:
        print(f" - {OUT_MISSING}（找不到 wav 的檔案共 {len(missing_wav)}）")
    print(f" - 已複製的 wav 存放於：{dapu_out_dir} 、{zhaoan_out_dir} 、{mix_out_dir}")

if __name__ == "__main__":
    main()
