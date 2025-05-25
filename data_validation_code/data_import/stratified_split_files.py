import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def stratified_split_and_move_files():
    """
    Read CSV, perform stratified split, and move files to test/train directories
    """
    
    # Hardcoded variables - modify as needed
    source_dir_original = "/g/data/eu59/data_flowmop/ANUDC_16/FCS_files/cells_panel/"  # Directory with .fcs files
    output_base_dir = "/g/data/eu59/data_flowmop/ANUDC_16/FCS_files/cells_stratified_split/"  # Base directory for test/train folders
    csv_path = "/g/data/eu59/data_flowmop/ANUDC_16/FCS_files/selected_metadata_anudc_columns.csv"
    label_column = "day_numeric"
    test_size = 0.2
    seed = 42

    # Read metadata CSV
    df = pd.read_csv(csv_path)
    
    # Perform stratified split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[label_column], 
        random_state=seed
    )
    
    # Create output directories
    train_dir = Path(output_base_dir) / "train"
    test_dir = Path(output_base_dir) / "test"
    
    for dir_path in [train_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Move original files
    move_files(train_df['filename'], source_dir_original, train_dir, "")
    move_files(test_df['filename'], source_dir_original, test_dir, "")
        
    print(f"Train set: {len(train_df)} files")
    print(f"Test set: {len(test_df)} files")
    print(f"Class distribution in train: {train_df[label_column].value_counts().to_dict()}")
    print(f"Class distribution in test: {test_df[label_column].value_counts().to_dict()}")

def move_files(filenames, source_dir, dest_dir, suffix):
    """
    Move files from source to destination directory
    """
    moved_count = 0
    missing_count = 0
    
    for filename in filenames:
        # Handle different file extensions
        base_name = Path(filename).stem
        if suffix:
            source_file = Path(source_dir) / f"{base_name}{suffix}.fcs"
        else:
            source_file = Path(source_dir) / filename
        
        if source_file.exists():
            dest_file = dest_dir / source_file.name
            shutil.move(str(source_file), str(dest_file))
            moved_count += 1
        else:
            print(f"Warning: File not found: {source_file}")
            missing_count += 1
    
    print(f"Moved {moved_count} files to {dest_dir}")
    if missing_count > 0:
        print(f"Missing {missing_count} files in {source_dir}")

if __name__ == "__main__":
    
    stratified_split_and_move_files()