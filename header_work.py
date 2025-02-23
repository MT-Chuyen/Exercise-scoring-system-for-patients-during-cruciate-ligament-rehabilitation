import pandas as pd
import math
import numpy as np
def read_csv_with_variable_columns(file_path):
    # Open the file using the provided file path
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
    
    return df

def find_page_positions(df):
    # Define page names
    page_names = ['Joints', 'Model Outputs', 'Segments', 'Trajectories']
    page_positions = {}

    # Find the 'Frame' index to locate the 'Devices' page
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
                if len(empty_string_positions) >= 3:
                    # Take the third occurrence as the end of the page if there are more than 3 empty strings
                    end_index = empty_string_positions[2] - 2
                    page_positions[page_name] = (start_index, end_index)
                elif len(empty_string_positions) == 2:
                    # If there are exactly 2 empty strings, set the end of the page to the end of the DataFrame
                    page_positions[page_name] = (start_index, len(df))
                else:
                    # If there are less than three empty strings found, set the end index to None
                    page_positions[page_name] = (start_index, None)

    return page_positions

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

def extract_vector(User_Name, page_positions, df, name_header, name_page, Plane_Sel, sel_part, reference_frame):
    """
    Extracts vectors in different 2D planes based on the selection.

    Parameters:
    - name_header: The header name to be used for identifying the column.
    - name_page: The name of the page where the data is located.
    - Plane_Sel: The plane selection ("YZ", "XZ", "XY", "X", "Y", "Z") to specify which columns to extract.
    - sel_part: The selected part of the header name.
    - reference_frame: The reference frame from which to extract the data.

    Returns:
    - A list containing the extracted vector(s) based on the Plane_Sel.
    """
    header_name = User_Name + sel_part + name_header
    page_start, page_end = page_positions[name_page]
    header_pos = find_header_in_row(df, header_name, page_start)

    # Depending on Plane_Sel, select the appropriate columns
    if Plane_Sel == "YZ":
        return [
            float(df.iloc[page_start + reference_frame + 2, header_pos + 1]),
            float(df.iloc[page_start + reference_frame + 2, header_pos + 2])
        ]
    elif Plane_Sel == "XZ":
        return [
            float(df.iloc[page_start + reference_frame + 2, header_pos]),
            float(df.iloc[page_start + reference_frame + 2, header_pos + 2])
        ]
    elif Plane_Sel == "XY":
        return [
            float(df.iloc[page_start + reference_frame + 2, header_pos]),
            float(df.iloc[page_start + reference_frame + 2, header_pos + 1])
        ]
    elif Plane_Sel == "X":
        return [
            float(df.iloc[page_start + reference_frame + 2, header_pos]),
        ]
    elif Plane_Sel == "Y":
        return [
            float(df.iloc[page_start + reference_frame + 2, header_pos + 1]),
        ]
    elif Plane_Sel == "Z":
        return [
            float(df.iloc[page_start + reference_frame + 2, header_pos + 2]),
        ]

#calculate angle between 2 vectors
def calculate_angle_between_vectors(vec1, vec2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitude (norm) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Calculate the cosine of the angle between the two vectors
    cos_angle = dot_product / (norm_vec1 * norm_vec2)

    # Calculate the angle in radians, then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def check_left_or_right(reference_vec, other_vec):
    # This function now only compares the x-coordinates of the vectors.
    if reference_vec[0] < other_vec[0]:
        return "right"   # "left" here means reference_vec is to the left of other_vec
    elif reference_vec[0] > other_vec[0]:
        return "left"  # "right" means reference_vec is to the right of other_vec
    else:
        return "aligned"  # "aligned" if the x-coordinates are the same