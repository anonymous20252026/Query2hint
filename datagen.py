import pandas as pd
import numpy as np
import ast
import os

def generate_reptile_datasets():
    # Define files
    files = {
        "JOB": "benchmark/dataset/job.csv",
        "CEB": "benchmark/dataset/ceb.csv"  # Make sure your big file is named this
    }
    
    for task_name, filename in files.items():
        if not os.path.exists(filename):
            print(f"⚠️  {filename} not found. Skipping {task_name}...")
            continue
            
        print(f"🔹 Processing {task_name} workload...")
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

        # 1. Parse string lists to actual python lists
        # We use a safe converter that handles potential parsing errors
        def safe_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else x
            except:
                return []

        df['runtime_list'] = df['runtime_list'].apply(safe_eval)
        df['hint_list'] = df['hint_list'].apply(safe_eval)
        
        # Calculate Mean Runtime
        df['mean_runtime'] = df['runtime_list'].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else 0)
        
        # Remove invalid rows
        df = df[df['mean_runtime'] > 0]

        triplets = []
        
        # 2. Group by SQL to finding Winners vs Losers
        grouped = df.groupby('sql')
        print(f"   Analyzing {len(grouped)} unique queries in {task_name}...")

        for sql_text, group in grouped:
            if len(group) < 2: continue # Need at least 2 runs to compare
                
            sorted_group = group.sort_values('mean_runtime')
            
            # WINNER (Positive)
            best_row = sorted_group.iloc[0]
            best_time = best_row['mean_runtime']
            
            # Convert [24] -> "Hint_24"
            pos_ids = sorted(best_row['hint_list'])
            pos_text = " ".join([f"Hint_{i}" for i in pos_ids]) if pos_ids else "Default_Plan"
            
            # LOSERS (Hard Negatives)
            # We enforce a 10% gap to ensure it's a real optimization difference
            for _, row in sorted_group.iloc[1:].iterrows():
                curr_time = row['mean_runtime']
                if curr_time > best_time * 1.10: 
                    neg_ids = sorted(row['hint_list'])
                    neg_text = " ".join([f"Hint_{i}" for i in neg_ids]) if neg_ids else "Default_Plan"
                    
                    triplets.append({
                        "anchor": sql_text,
                        "positive": pos_text,
                        "negative": neg_text,
                        "source_task": task_name,
                        "time_diff": curr_time - best_time
                    })
        
        # 3. Save
        out_file = f"stage1_triplets_{task_name}.csv"
        pd.DataFrame(triplets).to_csv(out_file, index=False)
        print(f"✅ Generated {len(triplets)} training samples for {task_name} -> {out_file}")

if __name__ == "__main__":
    generate_reptile_datasets()