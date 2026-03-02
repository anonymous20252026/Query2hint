import pandas as pd
import numpy as np
import ast

def generate_stage1_dataset(input_path="job.csv", output_path="stage1_contrastive_dataset.csv"):
    print(f"📂 Loading {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("❌ Error: job.csv not found. Please upload it.")
        return

    # 1. Parse String Lists to Actual Lists
    print("⚙️  Parsing data...")
    df['runtime_list'] = df['runtime_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['hint_list'] = df['hint_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Calculate Mean Runtime for stability
    df['mean_runtime'] = df['runtime_list'].apply(lambda x: np.mean(x) if isinstance(x, list) else x)

    triplets = []
    
    # 2. Group by SQL Query
    grouped = df.groupby('sql')
    print(f"📊 Processing {len(grouped)} unique SQL queries...")

    for sql_text, group in grouped:
        # Sort by Runtime: Fastest (Winner) -> Slowest (Loser)
        sorted_group = group.sort_values('mean_runtime')
        
        # --- Identify Positive (The Winner) ---
        best_row = sorted_group.iloc[0]
        best_time = best_row['mean_runtime']
        
        # Convert ID List [23, 45] -> Token String "Hint_23 Hint_45"
        # This creates a specific "Class Token" for the BERT model to learn.
        pos_ids = sorted(best_row['hint_list'])
        pos_text = " ".join([f"Hint_{i}" for i in pos_ids]) if pos_ids else "Default_Plan"
        
        # --- Identify Negatives (The Losers) ---
        # We look for plans that are significantly slower (>15% worse)
        # to ensure the model learns a meaningful distinction.
        for _, row in sorted_group.iloc[1:].iterrows():
            curr_time = row['mean_runtime']
            
            if curr_time > best_time * 1.15: 
                neg_ids = sorted(row['hint_list'])
                neg_text = " ".join([f"Hint_{i}" for i in neg_ids]) if neg_ids else "Default_Plan"
                
                # Add Triplet
                triplets.append({
                    "anchor": sql_text,      # The Query
                    "positive": pos_text,    # The Fast Hint
                    "negative": neg_text,    # The Slow Hint
                    "time_diff": curr_time - best_time # Metadata (optional)
                })

    # 3. Save to CSV
    out_df = pd.DataFrame(triplets)
    # Shuffle the dataset
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_df.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS! Generated {len(out_df)} triplets.")
    print(f"💾 Saved to: {output_path}")
    print("\n--- Sample Row ---")
    print(out_df.iloc[0])

if __name__ == "__main__":
    inputpath="benchmark/dataset/job.csv"
    outputpath="benchmark/dataset/stage1_contrastive_dataset.csv"
    generate_stage1_dataset(input_path=inputpath, output_path=outputpath)