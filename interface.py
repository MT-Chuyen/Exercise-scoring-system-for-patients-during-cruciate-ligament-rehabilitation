from singlehop import Single_Hop
from header_work import read_csv_with_variable_columns, find_page_positions
from triple_hop import calculate_triple_hop_distance
from less_test import less_score
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import xlsxwriter
import pandas as pd
def find_trial_csv(folder, trial_name):
    """Find a CSV file in a folder that contains the trial_name."""
    for file in os.listdir(folder):
        if trial_name in file and file.endswith(".csv"):
            return os.path.join(folder, file)
    return None


def process_files(folder_path, trial_names,user_name,bw):
    """
    Processes trial files based on test type and selected part (R, L, RL).
    For Single Hop Test and Triple Hop Test: Processes files for "R" and "L" when "RL" is specified.
    For LESS Test, it processes all files regardless of Sel_Part.
    """
    results = {}
    subdirs = ["LESS Test", "Single Hop Test", "Triple Hop Test"]
    side_conditions = {
        "R": ["Right"],
        "L": ["Left"],
        "RL": ["Right", "Left"]
    }
    Sel_Part = "RL"
    if Sel_Part == "RL":
        parts_to_process = ["R", "L"]
    else:
        parts_to_process = [Sel_Part]

    for part in parts_to_process:
        sides_to_include = side_conditions.get(part, ["Right", "Left"])
        for subdir in subdirs:
            dir_path = os.path.join(folder_path, subdir)
            if os.path.exists(dir_path):
                for trial_name in trial_names:
                    for file in os.listdir(dir_path):
                        # LESS Test condition: Process all files
                        if subdir == "LESS Test" and trial_name in file and file.endswith(".csv"):
                            process_condition = True
                        # Single and Triple Hop Test condition: Check side based on current part being processed
                        elif subdir != "LESS Test" and trial_name in file and file.endswith(".csv") and any(side in file for side in sides_to_include):
                            process_condition = True
                        else:
                            process_condition = False

                        if process_condition:
                            csv_path = os.path.join(dir_path, file)
                            df = read_csv_with_variable_columns(csv_path)  # Ensure pd is imported and this function is correctly set to read your CSVs
                            page_positions = find_page_positions(df)  # Placeholder for your actual function

                            # Processing logic based on the test type and selected part
                            if subdir == "Single Hop Test":
                                print(f"Single Hop Test {trial_name}" )
                                result = Single_Hop(df, user_name, part, "Force", page_positions, 30, bw)
                                results[f"{subdir} - {trial_name} - Part {part}"] = result
                            elif subdir == "Triple Hop Test":
                                print(f"Triple Hop test {trial_name}" )
                                try:
                                    result = calculate_triple_hop_distance(df, user_name, part)
                                    results[f"{subdir} - {trial_name} - Part {part}"] = result
                                except:
                                    results[f"{subdir} - {trial_name} - Part {part}"] = 0
                            elif subdir == "LESS Test":
                                # LESS Test logic remains unchanged
                                print(f"LESS test {trial_name}" )
                                try:
                                    result = less_score(df, user_name, page_positions, "3", "4", "Force", 30)
                                    results[f"{subdir} - {trial_name}"] = result
                                except:
                                    results[f"{subdir} - {trial_name}"] = [0]
    return results
def sanitize_filename(filename):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  # Add more if needed
    for char in invalid_chars:
        filename = filename.replace(char, '_')  # Replace invalid chars with underscore
    return filename

def process_tests():
    # Retrieve user input
    user_name = user_name_entry.get()
    user_name = user_name + ":"
    info = info_entry.get()
    bw = int(bw_entry.get())

    # Select folder
    folder_path = filedialog.askdirectory()
    if folder_path:
        # Validate the user input for Sel_Parts
        if True:
            trial_names = ["Trial1", "Trial2", "Trial3"]
            # Process files with the selected folder path, trial names, and the Sel_Part input
            results = process_files(folder_path, trial_names,user_name, bw)
            write_results_to_excel(results, user_name, info)

def write_results_to_excel(results,user_name, info):
    # Specify the path where the Excel file will be saved
    sanitized_user_name = sanitize_filename(user_name)

    excel_path = f'{sanitized_user_name}_results.xlsx'

    # Create a new Excel file and add a worksheet
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet('Test Results')

    # Formats for merged headers and sub-headers
    merge_format = workbook.add_format({
        'align': 'center',
        'valign': 'vcenter',
        'bold': True
    })
    header_format = workbook.add_format({'bold': True})

    # Add main test type titles and merge the cells for them
    worksheet.merge_range('A4:C4', 'Single Leg Hop Test', merge_format)
    worksheet.merge_range('A16:C16', 'Triple Leg Hop Test', merge_format)
    worksheet.write('A27', 'LESS Test', merge_format)

    # Add column headers for the Single Leg Hop Test
    headers = ['Side', 'Trial', 'Limb stability', '', 'Pelvis stability', 'Trunk stability', 'Shock absorption', 'Movement strategy']
    worksheet.write_row('A5', headers, header_format)

    # Merge cells for 'Limb stability' header
    worksheet.merge_range('C5:D5', 'Limb stability', merge_format)

    # Add sub-headers for 'Limb stability'
    sub_headers = ['FPKPA', 'GRF vs KJC']
    worksheet.write_row('C6', sub_headers, header_format)

    # Set the column widths to accommodate the headers
    worksheet.set_column('A:A', 10)  # Side column
    worksheet.set_column('B:B', 10)  # Trial column
    worksheet.set_column('C:D', 15)  # Limb stability columns
    worksheet.set_column('E:E', 20)  # Pelvis stability column
    worksheet.set_column('F:F', 20)  # Trunk stability column
    worksheet.set_column('G:G', 20)  # Shock absorption column
    worksheet.set_column('H:H', 20)  # Movement strategy column
    worksheet.set_column('I:I', 20)  # Movement strategy column
    worksheet.set_column('J:J', 10)  # Movement strategy column
    worksheet.set_column('K:K', 10)  # Movement strategy column

    # Add specified text to the cells
    worksheet.merge_range('A1:C1', user_name,merge_format)
    worksheet.merge_range('A2:C2', info,merge_format)
    worksheet.write('A7', 'Left')
    worksheet.write('A11', 'Right')

    worksheet.write('A18', 'Left')
    worksheet.write('A22', 'Right')

    worksheet.write_row('A17', ['Side', 'Trial', 'Distance (mm)'], header_format)
    points = ['Point A', 'Point B', 'Point C', 'Point D', 'Point E', 
          'Point F', 'Point G', 'Point H', 'Point I', 'Point J', 
          'Point K', 'Point L', 'Point M', 'Point N', 'Point O', 'Point P', 'Point Q']

    # Prepend 'Trial' to the list of points
    row_data = ['Trial'] + points

    # Write the row to the worksheet starting at cell A28
    worksheet.write_row('A28', row_data, header_format)

    # Writing trial numbers 1, 2, 3, and "Average" for left and right
    for i in range(3):
        row = 6 + i
        worksheet.write(row, 1, i+1)

    for i in range(3):
        row = 10 + i
        worksheet.write(row, 1, i+1)

    for i in range(4):  # For trials 1 to 3 and "Average L"
        row = 17 + i
        worksheet.write(row, 1, i+1 if i < 3 else "Average L")

    for i in range(4):  # For trials 1 to 3 and "Average R"
        row = 21 + i
        worksheet.write(row, 1, i+1 if i < 3 else "Average R")

    # Writing trial numbers 1, 2, 3 for LESS
    for i in range(3):
        row = 28 + i
        worksheet.write(row, 0, i+1)

    columns = {
        'FPKPA': 'C',
        'GRF and KJC': 'D',
        'Pelvis Stability': 'E',
        'Trunk Stability': 'F',
        'Shock Absorption': 'G',
        'Movement Strategy': 'H'
    }

    columns_less = {}

    # Starting ASCII value for column 'B' (since 'A' is 65, 'B' is 66)
    start_column = 66  # ASCII value for 'B'

    # Generate column letters for each point from 'Point A' to 'Point O'
    for i in range(15):  # There are 15 points from A to O
        point_label = f"Point {chr(ord('A') + i)}"  # Create label like 'Point A', 'Point B', etc.
        columns_less[point_label] = chr(start_column + i)  # Map to corresponding Excel column
    
    for i in range(1, 4):
        row = str(28 + i)  # Rows 29,30,31 for LESS case
        for metric, col in columns_less.items():
            key = f'LESS Test - Trial{i}'
            if key in results and metric in results[key]:
                worksheet.write(f"{col}{row}", results[key][metric])
        
    # Write values for Left side, trials 1 to 3
    for i in range(1, 4):
        row = str(6 + i)  # Rows 7, 8, 9 for Left side
        for metric, col in columns.items():
            key = f'Single Hop Test - Trial{i} - Part L'
            if key in results and metric in results[key]:
                worksheet.write(f"{col}{row}", results[key][metric])

    worksheet.merge_range('K5:L5', "Jump Distance (mm)", merge_format)

# Write the function to print the value of Jump Distance in Single Hop 

    columns_distance = {
        'Distance': 'K',
    }

    for i in range(1, 4):
        row = str(6 + i)  # Rows 7, 8, 9 for Left side
        for metric, col in columns_distance.items():
            key = f'Single Hop Test - Trial{i} - Part L'
            if key in results and metric in results[key]:
                worksheet.write(f"{col}{row}", results[key][metric])

    # Calculate the average for the left side and write it to the cell
    left_values_single = [results[f'Single Hop Test - Trial{i} - Part L']['Distance'] for i in range(1, 4)]

    average_left_single = sum(left_values_single) / len(left_values_single)
    worksheet.write("K10", int(average_left_single))  # Writing the average to C21 for the left side
    worksheet.write("J10", "AverageL")
    worksheet.write("J14", "AverageR") 
    for i in range(1, 4):
        row = str(10 + i)  # Rows 11, 12, 13 for Right side
        for metric, col in columns_distance.items():
            key = f'Single Hop Test - Trial{i} - Part R'
            if key in results and metric in results[key]:
                worksheet.write(f"{col}{row}", results[key][metric])
    # Calculate the average for the right side and write it to the cell
    right_values_single = [results[f'Single Hop Test - Trial{i} - Part R']['Distance'] for i in range(1, 4)]
    average_right_single = sum(right_values_single) / len(right_values_single)
    worksheet.write("K14", int(average_right_single))  # Writing the average to C25 for the right side

    # Write values for Right side, trials 1 to 3
    for i in range(1, 4):
        row = str(10 + i)  # Rows 11, 12, 13 for Right side
        for metric, col in columns.items():
            key = f'Single Hop Test - Trial{i} - Part R'
            if key in results and metric in results[key]:
                worksheet.write(f"{col}{row}", results[key][metric])
    '''
    # Writing values for Triple Hop Test - Left Side
    for i in range(1, 4):
        key = f'Triple Hop Test - Trial{i} - Part L'
        if key in results:
            # Row numbers are 18, 19, 20 for the left side
            worksheet.write(f"C{17 + i}", results[key])

    # Calculate the average for the left side and write it to the cell
    left_values = [results[f'Triple Hop Test - Trial{i} - Part L'] for i in range(1, 4)]
    average_left = sum(left_values) / len(left_values)
    worksheet.write("C21", average_left)  # Writing the average to C21 for the left side

    # Writing values for Triple Hop Test - Right Side
    for i in range(1, 4):
        key = f'Triple Hop Test - Trial{i} - Part R'
        if key in results:
            # Row numbers are 22, 23, 24 for the right side
            worksheet.write(f"C{21 + i}", results[key])

    # Calculate the average for the right side and write it to the cell
    right_values = [results[f'Triple Hop Test - Trial{i} - Part R'] for i in range(1, 4)]
    average_right = sum(right_values) / len(right_values)
    worksheet.write("C25", average_right)  # Writing the average to C25 for the right side

    if average_right > average_left:
        different = average_left/average_right * 100
    else:
        different = average_right/ average_left *100
    int_label = 100 - int(different) 
    label_different = str(int_label) + "%"
    
    worksheet.merge_range('E17:F17', "Percentage difference between two legs",merge_format)
    worksheet.merge_range('E18:F18', label_different,merge_format)
    '''
    # Writing values for LESS Test

    worksheet.write('I5', 'Total Score', header_format)

    # Calculate and write the total scores for the left sid
    for i in range(7, 10):  # Rows 7 to 9 for the left side
        worksheet.write_formula(f'I{i}', f'=SUM(C{i}:H{i})')

    # Calculate and write the total scores for the right side
    for i in range(11, 14):  # Rows 11 to 13 for the right side
        worksheet.write_formula(f'I{i}', f'=SUM(C{i}:H{i})')
    # Close the workbook to save the Excel file
    workbook.close()

# Set up the UI
root = tk.Tk()
root.title("Test Processing System")

# Define and place the entry widgets and their labels
labels_texts = [
    "User Name:", "Body Weight (grams):", "Info:"
]
examples = [
    "Example: 'Le Duc Manh ACL'","Example: '665'", "Give the situation of patient"
]

# Dynamically create and place labels, entries, and example labels
for i, (label_text, example) in enumerate(zip(labels_texts, examples)):
    tk.Label(root, text=label_text).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    tk.Label(root, text=example).grid(row=i, column=2)
    # Assign each entry widget to a global variable for later access
    if label_text.startswith("User Name"):
        user_name_entry = entry
 
    elif label_text.startswith("Body Weight"):
        bw_entry = entry

    elif label_text.startswith("Info"):
        info_entry = entry
    

# Add a button to trigger processing
process_button = tk.Button(root, text="Process Tests", command=process_tests)
process_button.grid(row=len(labels_texts), column=1)

root.mainloop()