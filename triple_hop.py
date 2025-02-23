from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from header_work import find_header_in_row,  calculate_angle_between_vectors, check_left_or_right

def calculate_triple_hop_distance(df, User_Name, Sel_Part):
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

    # Assuming the peaks are already sorted by their height, take the first three
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
        point_end = sel_trajectories_array[min_value_index][0]
        vector = point_end - point_start
        projection_length = np.dot(vector, [1, 0, 0])
        print(f"The distance of Triple Hop Test in {Sel_Part} leg is: {projection_length} mm")
        return int(projection_length)
    else:
        print("Insufficient data to calculate the projection length.")
        return None