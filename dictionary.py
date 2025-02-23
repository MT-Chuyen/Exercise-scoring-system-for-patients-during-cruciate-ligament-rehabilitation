import pandas as pd
import xlsxwriter
user_name = "Le Duc Manh ACL:"
results = {'LESS Test - Trial1': 0, 'LESS Test - Trial2': 0, 'LESS Test - Trial3': 0, 'Single Hop Test - Trial1 - Part R': {'FPKPA': 2, 'GRF and KJC': 0, 'Pelvis Stability': 2, 'Trunk Stability': 2, 'Shock Absorption': 0, 'Movement Strategy': 0}, 'Single Hop Test - Trial2 - Part R': {'FPKPA': 2, 'GRF and KJC': 0, 'Pelvis Stability': 2, 'Trunk Stability': 2, 'Shock Absorption': 0, 'Movement Strategy': 0}, 'Single Hop Test - Trial3 - Part R': {'FPKPA': 2, 'GRF and KJC': 0, 'Pelvis Stability': 2, 'Trunk Stability': 2, 'Shock Absorption': 0, 'Movement Strategy': 0}, 'Triple Hop Test - Trial1 - Part R': 4700.666382, 'Triple Hop Test - Trial2 - Part R': 4627.062255000001, 'Triple Hop Test - Trial3 - Part R': 4391.2229, 'Single Hop Test - Trial1 - Part L': {'FPKPA': 0, 'GRF and KJC': 1, 'Pelvis Stability': 2, 'Trunk Stability': 2, 'Shock Absorption': 1, 'Movement Strategy': 1}, 'Single Hop Test - Trial2 - Part L': {'FPKPA': 0, 'GRF and KJC': 1, 'Pelvis Stability': 2, 'Trunk Stability': 2, 'Shock Absorption': 1, 'Movement Strategy': 1}, 'Single Hop Test - Trial3 - Part L': {'FPKPA': 0, 'GRF and KJC': 1, 'Pelvis Stability': 2, 'Trunk Stability': 2, 'Shock Absorption': 1, 'Movement Strategy': 1}, 'Triple Hop Test - Trial1 - Part L': 5515.828858000001, 'Triple Hop Test - Trial2 - Part L': 4973.448243, 'Triple Hop Test - Trial3 - Part L': 4711.859375}

# Specify the path where the Excel file will be saved
excel_path = 'Test_Results.xlsx'

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
worksheet.write('A27', 'LESS', merge_format)

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

# Add specified text to the cells
worksheet.merge_range('A1:C1', user_name,merge_format)
worksheet.write('A7', 'Left')
worksheet.write('A11', 'Right')

worksheet.write('A18', 'Left')
worksheet.write('A22', 'Right')

worksheet.write_row('A17', ['Side', 'Trial', 'Distance (mm)'], header_format)
worksheet.write_row('A28', ['Trial', 'Score'], header_format)

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

# Write values for Left side, trials 1 to 3
for i in range(1, 4):
    row = str(6 + i)  # Rows 7, 8, 9 for Left side
    for metric, col in columns.items():
        key = f'Single Hop Test - Trial{i} - Part L'
        if key in results and metric in results[key]:
            worksheet.write(f"{col}{row}", results[key][metric])

# Write values for Right side, trials 1 to 3
for i in range(1, 4):
    row = str(10 + i)  # Rows 11, 12, 13 for Right side
    for metric, col in columns.items():
        key = f'Single Hop Test - Trial{i} - Part R'
        if key in results and metric in results[key]:
            worksheet.write(f"{col}{row}", results[key][metric])

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

# Writing values for LESS Test
for i in range(1, 4):
    key = f'LESS Test - Trial{i}'
    if key in results:
        # Row numbers are 29, 30, 31 for the LESS Test
        worksheet.write(f"B{28 + i}", results[key])
worksheet.write('I5', 'Total Score', header_format)

# Calculate and write the total scores for the left side
for i in range(7, 10):  # Rows 7 to 9 for the left side
    worksheet.write_formula(f'I{i}', f'=SUM(B{i}:H{i})')

# Calculate and write the total scores for the right side
for i in range(11, 14):  # Rows 11 to 13 for the right side
    worksheet.write_formula(f'I{i}', f'=SUM(B{i}:H{i})')
# Close the workbook to save the Excel file
workbook.close()