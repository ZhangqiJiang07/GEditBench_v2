import pandas as pd
import pyarrow.parquet as pq
from megfile import smart_open, smart_glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import os
import threading
from collections import defaultdict

from common_utils.project_paths import DATA_ROOT

MAX_WORKERS = os.cpu_count()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path-to-uniedit-data', type=str,
        default="path/to/XrZUMXM-/xiaotanhua/UnicEdit-10M/data",
        help='Path to save the prepared UniEdit data.'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=str(DATA_ROOT / "a_raw_img_prompt_pair_data" / "UnicEdit-10M"),
        help='Path to save the raw data.'
    )
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Maximum number of worker threads for processing.')
    return parser.parse_args()


class JsonlWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.file_handles = {}
        self.lock = threading.Lock()

    def write_batch(self, subtask_name, records):
        if not subtask_name:
            subtask_name = "unknown_subtask"
        safe_filename = str(subtask_name).replace(" ", "_").replace("/", "_").replace("\\", "_") + ".jsonl"
        file_path = os.path.join(self.output_dir, safe_filename)

        with self.lock:
            if file_path not in self.file_handles:
                self.file_handles[file_path] = open(file_path, 'a', encoding='utf-8')
            
            f = self.file_handles[file_path]
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close_all(self):
        for f in self.file_handles.values():
            f.close()

def process_single_parquet(file_path):
    grouped_data = defaultdict(list)
    try:
        with smart_open(file_path, 'rb') as f:
            pfile = pq.ParquetFile(f)
            target_columns = ['key', 'edit_subtask', 'prompt_en']

            current_row_idx = 0
            for i in range(pfile.num_row_groups):
                table = pfile.read_row_group(i, columns=target_columns)
                df = table.to_pandas()
                
                for key, subtask, prompt in zip(df['key'], df['edit_subtask'], df['prompt_en']):
                    
                    record = {
                        "key": str(key),
                        "prompt_en": str(prompt) if prompt else "",
                        "parquet_file": file_path,
                        "row_idx": current_row_idx 
                    }
                    
                    task_key = subtask if subtask else "Uncategorized"
                    grouped_data[task_key].append(record)
                    
                    current_row_idx += 1
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    return grouped_data

def main():
    args = parse_args()
    print(f"Scaning {args.path_to_uniedit_data} ...")
    all_files = list(smart_glob(f"{args.path_to_uniedit_data}/**/*.parquet", recursive=True))
    
    if not all_files:
        print("No Parquet file found, please check the path!")
        return

    print(f"Found {len(all_files)} Parquet files. Starting multi-threaded processing...")

    writer = JsonlWriter(args.output_dir)
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_single_parquet, path): path for path in all_files}
            
            for future in tqdm(as_completed(futures), total=len(all_files), desc="Processing"):
                batch_result = future.result()
                
                if batch_result:
                    for subtask, records in batch_result.items():
                        writer.write_batch(subtask, records)
                        
    finally:
        print("\nClosing file handles...")
        writer.close_all()
        print(f"Finished! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
