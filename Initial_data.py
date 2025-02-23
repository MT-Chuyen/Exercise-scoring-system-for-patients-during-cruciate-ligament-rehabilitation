import os
import shutil
import tkinter as tk
from tkinter import filedialog

def rename_and_move_files(folder_path):
    less_test_path = 'E:/Scoring System/LESS Test'
    single_hop_test_path = 'E:/Scoring System/Single Hop Test'
    triple_hop_test_path = 'E:/Scoring System/Triple Hop Test'

    os.makedirs(less_test_path, exist_ok=True)
    os.makedirs(single_hop_test_path, exist_ok=True)
    os.makedirs(triple_hop_test_path, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    trial_less = trial_single_left = trial_single_right = trial_triple_left = trial_triple_right = 1

    for filename in files:
        old_path = os.path.join(folder_path, filename)
        original_name = os.path.splitext(filename)[0]  # Get the original name without extension
        new_name = ""
        new_path = ""

        if 'LESS' in filename:
            new_name = f"Trial{trial_less}_{original_name}.csv"
            new_path = os.path.join(less_test_path, new_name)
            trial_less += 1
        elif 'singleL' in filename:
            new_name = f"Trial{trial_single_left}_{original_name}_Left.csv"
            new_path = os.path.join(single_hop_test_path, new_name)
            trial_single_left += 1
        elif 'singleR' in filename:
            new_name = f"Trial{trial_single_right}_{original_name}_Right.csv"
            new_path = os.path.join(single_hop_test_path, new_name)
            trial_single_right += 1
        elif 'tripleL' in filename:
            new_name = f"Trial{trial_triple_left}_{original_name}_Left.csv"
            new_path = os.path.join(triple_hop_test_path, new_name)
            trial_triple_left += 1
            
        elif 'tripleR' in filename:
            new_name = f"Trial{trial_triple_right}_{original_name}_Right.csv"
            new_path = os.path.join(triple_hop_test_path, new_name)
            trial_triple_right += 1

        if new_name:  # Ensure there is a new name before moving
            shutil.move(old_path, new_path)
            print(f"Moved '{old_path}' to '{new_path}'")

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        rename_and_move_files(folder_selected)

select_folder()
