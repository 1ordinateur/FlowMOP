#!/usr/bin/env python
"""
Utility to visualize profiling results using snakeviz.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
import subprocess
import time

def check_snakeviz_installed():
    """Check if snakeviz is installed."""
    try:
        import snakeviz
        return True
    except ImportError:
        return False

def install_snakeviz():
    """Install snakeviz if missing."""
    print("SnakeViz is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "snakeviz"])
    print("SnakeViz installed successfully.")

def visualize_profile(profile_file):
    """Launch snakeviz to visualize a profile file."""
    profile_path = Path(profile_file)
    if not profile_path.exists():
        print(f"Error: Profile file '{profile_file}' does not exist.")
        return False

    # Make sure snakeviz is installed
    if not check_snakeviz_installed():
        install_snakeviz()
    
    # Launch snakeviz to visualize profile
    print(f"Launching SnakeViz for profile: {profile_file}")
    subprocess.Popen([sys.executable, "-m", "snakeviz", str(profile_path)])
    
    # Give it a moment to launch
    time.sleep(1)
    
    return True

def list_profile_files(profile_dir):
    """List all profile files in the given directory."""
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        print(f"Error: Profile directory '{profile_dir}' does not exist.")
        return []
    
    profile_files = list(profile_dir.glob("*.prof"))
    if not profile_files:
        print(f"No profile files found in '{profile_dir}'.")
        return []
    
    print(f"Found {len(profile_files)} profile files:")
    for i, file in enumerate(profile_files, 1):
        print(f"{i}. {file.name}")
    
    return profile_files

def main():
    parser = argparse.ArgumentParser(description='Visualize Python profiling results with SnakeViz')
    parser.add_argument('--profile-file', help='Path to the profile file to visualize')
    parser.add_argument('--profile-dir', default='./profile_results', 
                        help='Directory containing profile files')
    
    args = parser.parse_args()
    
    if args.profile_file:
        visualize_profile(args.profile_file)
    else:
        profile_files = list_profile_files(args.profile_dir)
        if profile_files:
            print("\nSelect a profile file to visualize (enter the number):")
            try:
                selection = int(input("> "))
                if 1 <= selection <= len(profile_files):
                    visualize_profile(profile_files[selection-1])
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(profile_files)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

if __name__ == '__main__':
    main() 