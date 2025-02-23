import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from header_work import find_header_in_row,  calculate_angle_between_vectors, check_left_or_right
from scipy.signal import find_peaks
#def Single_Hop(df, User_Name, Sel_Part, kinetic_sel, page_positions, thres_hold_init_cont, bw)
import os
import csv
import numpy as np
import pandas as pd
import sys
import matplotlib as plt
import seaborn as sns
import math
import matplotlib.pyplot as plt
from header_work import find_header_in_row,  calculate_angle_between_vectors, check_left_or_right
from singlehop import Single_Hop
from triple_hop import calculate_triple_hop_distance
from scipy.signal import find_peaks
#User input
#User_Name = "Le Duc Manh ACL:" #Option: "Name with space:"
User_Name = "Dao Thi Hong Cham:" #Option: "Name with space:"
Sel_Part = "L"
Kinetic_Sel = "Force" #Option: "Force", "Moment", "CoP"
BW = 665
other_part = 0
Thres_Hold_Init_Cont = 30
Total_Score = 0
other_part = ""
kinetic_sel = "Force"
#File Location

#file_path = 'C:/Users/Admin/Desktop/Scoring System/Le Duc Manh/Single Hop Test - Copy/Trial1_Single_hop_Test_Left.csv'


file_path = 'E:/Scoring System/Triple Hop Test/Trial3_Cham TripleL03_Left.csv'
scores = {}

#read csv file
import pandas as pd
import numpy as np
with open(file_path, 'r', encoding='utf-8-sig') as file:
    lines = file.readlines()

# Split each line by comma, considering the possibility of varying number of columns
data = [line.strip().split(',') for line in lines if line.strip()]  # This also filters out empty lines

# Find the maximum length (number of columns) across all rows to standardize DataFrame construction
max_columns = max(len(row) for row in data)

# Ensure all rows have the same number of columns, padding shorter rows with empty strings
data_uniform = [row + [''] * (max_columns - len(row)) for row in data]

# Extract headers and data
headers = data_uniform[0]  # Assuming the first row contains headers
data_rows = data_uniform[1:]  # The rest are data rows

# Create DataFrame
df = pd.DataFrame(data_rows, columns=headers)

######## Define page names positions dictionary
page_names = ['Joints', 'Model Outputs','Segments', 'Trajectories']
page_positions = {}
#find the Devices page
frame_index = df.index[df.iloc[:, 0] == 'Frame'].tolist()[0] if df.index[df.iloc[:, 0] == 'Frame'].tolist() else None
if frame_index is not None:
            # Find positions with an empty string after the 'Frame' index
            empty_string_positions = df.iloc[frame_index:][df.iloc[frame_index:, 0] == ''].index.tolist()

            if len(empty_string_positions) >= 2:
                # If there are at least two empty strings, take the second one and subtract 1 from its index
                end_index_devices = empty_string_positions[1] - 2
            else:
                end_index_devices = None

devices_dict = {'Devices': (1, end_index_devices)}
# Combine 'Devices' dictionary with 'page_positions' dictionary
page_positions.update(devices_dict)
# Iterate through each page name and find the corresponding start and end index
for page_name in page_names:
    # Get the index for rows that contain the page name
    page_indices = df.index[df.iloc[:, 0] == page_name].tolist()
    for page_index in page_indices:
        start_index = page_index + 2  # Start index is two rows after the page name
        if start_index < len(df):
            # Find positions with an empty string after the start index
            empty_string_positions = df.iloc[start_index:][df.iloc[start_index:, 0] == ''].index.tolist()
            if start_index < len(df):
            # Find positions with an empty string after the start index
                empty_string_positions = df.iloc[start_index:][df.iloc[start_index:, 0] == ''].index.tolist()
            if len(empty_string_positions) >= 3:
                # Take the third occurrence as the end of the page if there are more than 3 empty strings
                end_index = empty_string_positions[2] - 2
                page_positions[page_name] = (start_index, end_index)
            elif len(empty_string_positions) == 2:
                # If there are exactly 3 empty strings, set the end of the page to the end of the DataFrame
                page_positions[page_name] = (start_index, len(df))
            else:
                # If there are less than three empty strings found, set the end index to None
                page_positions[page_name] = (start_index, None)

# Output the dictionary with the page start and end positions
print(page_positions)

def find_header_in_row(df, header_name,page_start):
  row = df.iloc[page_start,:]
  row = row.reset_index(drop=True)
  contains_value = row.str.contains(header_name, na=False, regex=False)

# Check if there is at least one match
  if contains_value.any():
    # Get the index of the first True value
    location_of_value = contains_value.idxmax()
    print(f"The column of {header_name}: {location_of_value}")
  else:
    print(f"The value {header_name} was not found.")
  return location_of_value

if Sel_Part == "R":
    target_plates = "1"
elif Sel_Part == "L":
    target_plates = "2"

page_start = 1
header_name_Sel = User_Name + Sel_Part + "HEE"
header_pos_Sel = find_header_in_row(df, header_name_Sel, page_start)
col_Sel = [header_pos_Sel, header_pos_Sel + 1, header_pos_Sel + 2]
col_Sel_Z = [header_pos_Sel + 2]

Sel_Trajectories = df.iloc[4:, col_Sel].reset_index(drop=True)
Sel_Trajectories_Z = df.iloc[4:, col_Sel_Z].reset_index(drop=True)

sel_trajectories_array = Sel_Trajectories.apply(pd.to_numeric, errors='coerce').to_numpy().reshape(-1, 1, 3)
sel_trajectories_Z_array = np.array(Sel_Trajectories_Z).flatten()

# Find the first index where the element is not an empty string
non_empty_indices = np.argwhere(sel_trajectories_Z_array != '').flatten()
if non_empty_indices.size > 0:
    first_non_empty_index = non_empty_indices[0]
else:
    first_non_empty_index = None  # In case all values are empty strings

# Replace empty strings with 0 and convert to float
sel_trajectories_Z_array = np.where(sel_trajectories_Z_array != '', sel_trajectories_Z_array, '0').astype(float)

# Calculate the height threshold as max value - 100
if float(np.max(sel_trajectories_Z_array)) > 600:
    threshold = 350
elif float(np.max(sel_trajectories_Z_array)) > 500:
    threshold = 300
elif float(np.max(sel_trajectories_Z_array)) > 400:
    threshold = 250
elif float(np.max(sel_trajectories_Z_array)) > 300:
    threshold = 200
elif float(np.max(sel_trajectories_Z_array)) > 200:
    threshold = 150
else:
    threshold = 100

height_threshold = np.max(sel_trajectories_Z_array) - threshold

# Find all peaks in the data above the height threshold
peaks, properties = find_peaks(sel_trajectories_Z_array, height=height_threshold, distance=40)
peak_values = properties['peak_heights']

first_three_peaks = peaks[:3]
first_three_peak_values = peak_values[:3]

# Ensure there is a third peak and calculate the minimum value in the window after the third peak
if len(peaks) >= 3:
    window_start = peaks[2]
    window_end = min(window_start + 100, len(sel_trajectories_Z_array))
    min_value_index = np.argmin(sel_trajectories_Z_array[window_start:window_end]) + window_start
    min_value = sel_trajectories_Z_array[min_value_index]

# Plot for debug (if needed)
# Plotting the time series and the first three peaks
plt.figure(figsize=(10, 6))
plt.plot(sel_trajectories_Z_array, marker='o', linestyle='-', color='b')
plt.plot(first_three_peaks, first_three_peak_values, "v", markersize=10, label='First three peaks', color='r')
if len(peaks) >= 3:
    plt.plot(min_value_index, min_value, "s", markersize=10, label='Minimum after third peak', color='magenta')
plt.title('Time Series Data of sel_trajectories_Z_array with First Three Peaks Above Threshold')
plt.xlabel('Time Frame')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()

if first_non_empty_index is not None and len(peaks) >= 3:
    # Calculate the distance
    point_start = sel_trajectories_array[first_non_empty_index][0]
    print(first_three_peaks[0])
    point_end = sel_trajectories_array[min_value_index][0]
    vector = point_end - point_start
    projection_length = np.dot(vector, [1, 0, 0])
    print(f"The distance of Triple Hop Test in {Sel_Part} leg is: {projection_length} mm")
else:
    print("Insufficient data to calculate the projection length.")