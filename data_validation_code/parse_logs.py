import argparse
import csv
from pathlib import Path
import re
import ast

def parse_log_file(log_file_path: Path):
    """
    Parses a single training.log file to extract metrics.
    It takes the last found values for each metric within the last training run.
    """
    last_epoch_stats = {}
    testing_stats = {}

    try:
        content = log_file_path.read_text()

        # Split the log file by '**** Start Training ****' to handle multiple runs
        training_runs = content.split('**** Start Training ****')
        if not training_runs:
            return None

        # Process the first training run
        first_run_content = training_runs[0]

        # Find the last epoch stats before 'Training Finished'
        # To handle cases where 'Training Finished' might be missing, we find all epoch stats
        # and take the last one.
        epoch_stats_matches = re.findall(r"Epoch stats: (\{.*\})", first_run_content)
        if epoch_stats_matches:
            try:
                # The last match is what we need
                stats_dict = ast.literal_eval(epoch_stats_matches[-1])
                last_epoch_stats['loss'] = stats_dict.get('loss')
                last_epoch_stats['val_loss'] = stats_dict.get('val_loss')
                last_epoch_stats['val_acc'] = stats_dict.get('val_acc')
                last_epoch_stats['val_auc'] = stats_dict.get('val_auc')
            except (SyntaxError, ValueError):
                pass  # Ignore malformed dictionaries

        # Find testing stats
        testing_stats_match = re.search(r'Testing Acc: ([\d.eE-]+), Testing Auc: ([\d.eE-]+)', first_run_content)
        if testing_stats_match:
            try:
                testing_stats['testing_acc'] = float(testing_stats_match.group(1))
                testing_stats['testing_auc'] = float(testing_stats_match.group(2))
            except ValueError:
                pass # Ignore conversion errors

        if not last_epoch_stats and not testing_stats:
            return None

        return {**last_epoch_stats, **testing_stats}

    except Exception as e:
        print(f"Error reading or processing file {log_file_path}: {e}")
        return None

def main(root_dir: str, output_file: str):
    """
    Finds all training.log files, parses them, and writes results to a CSV.
    """
    root_path = Path(root_dir)
    log_files = list(root_path.rglob('training.log'))

    if not log_files:
        print(f"No 'training.log' files found in '{root_dir}' and its subdirectories.")
        return

    results = []
    for log_file in log_files:
        data = parse_log_file(log_file)
        if data:
            experiment_name = log_file.parent.name
            result_row = {'experiment': experiment_name, **data}
            results.append(result_row)
            print(f"Parsed {log_file}")

    if not results:
        print("No data could be extracted from any log file.")
        return

    # Determine all possible fieldnames from the results to handle missing values gracefully
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Define the desired order of columns
    fieldnames_order = ['experiment', 'testing_auc', 'testing_acc', 'val_auc', 'val_acc', 'val_loss', 'loss']
    # Filter and order the found keys
    sorted_fieldnames = [f for f in fieldnames_order if f in all_keys]
    # Add any other keys that might have been found but are not in the predefined order
    sorted_fieldnames.extend(sorted([k for k in all_keys if k not in sorted_fieldnames]))


    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Results successfully written to {output_file}")
    print(f"Processed {len(log_files)} log files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse training logs and aggregate results into a CSV file.")
    parser.add_argument('root_dir', type=str, help="The root directory to search for training.log files (e.g., 'exp').")
    parser.add_argument('--output', type=str, default='training_results.csv', help="The path to the output CSV file.")
    args = parser.parse_args()

    main(args.root_dir, args.output) 