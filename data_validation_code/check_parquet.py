import pandas as pd
import os

print("Starting parquet file check...")
file_path = "./time_gate_outputs/concat_output_20032025.parquet"

print(f"Checking if file exists: {os.path.exists(file_path)}")
try:
    print("Attempting to read parquet file...")
    df = pd.read_parquet(file_path)
    print(f"Success! File can be read.")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()[:5]}...")
    
    # Check if sample ID columns exist
    if 'Sample_ID_Int' in df.columns and 'Sample_ID' in df.columns:
        print("\nSample ID columns found!")
        
        # Get value counts to show distribution of samples
        sample_int_counts = df['Sample_ID_Int'].value_counts().sort_index()
        print(f"\nSample_ID_Int value counts:")
        print(sample_int_counts)
        
        sample_id_counts = df['Sample_ID'].value_counts()
        print(f"\nSample_ID value counts:")
        print(sample_id_counts)
        
        # Show a few rows to confirm both columns exist
        print("\nSample of rows:")
        sample_cols = ['Sample_ID_Int', 'Sample_ID', 'Time']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(5))
    else:
        missing_cols = []
        if 'Sample_ID_Int' not in df.columns:
            missing_cols.append('Sample_ID_Int')
        if 'Sample_ID' not in df.columns:
            missing_cols.append('Sample_ID')
        print(f"Warning: Sample ID columns missing: {missing_cols}")
except Exception as e:
    print(f"Error reading file: {str(e)}")

print("Script completed.") 