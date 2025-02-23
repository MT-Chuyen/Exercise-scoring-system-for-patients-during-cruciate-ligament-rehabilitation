import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from header_work import find_header_in_row,  calculate_angle_between_vectors, check_left_or_right
from scipy.signal import find_peaks
def Single_Hop(df, User_Name, Sel_Part, kinetic_sel, page_positions, thres_hold_init_cont, bw):
    if Sel_Part == "R":
        target_plates = "1"
    elif Sel_Part == "L":
        target_plates = "2"
    name_plates = 'Imported AMTI 400600     #' + target_plates + ' - ' + kinetic_sel
    page_start, page_end = page_positions['Devices']

    # Find initial contact
    header_pos = find_header_in_row(df, name_plates, page_start)
    thres_frame = None
    print(page_start, page_end)
    for i in range(page_start + 3, page_end):
        try:
            current_value = math.sqrt(float(df.iloc[i, header_pos])**2 +
                                    float(df.iloc[i, header_pos+1])**2 +
                                    float(df.iloc[i, header_pos+2])**2)
            
            if current_value > 30:
                print(f"Initial Contact: {current_value} at index {i}")
                thres_frame = int(df.iloc[i, 0])
                print(thres_frame)
                break
        except ValueError:
            pass

    # Find max GRF
    max_value = -float('inf')  # Start with the smallest possible value
    max_position = None  # To store the position (row index) of the max value

    for i in range(page_start + 3, page_end):
        current_value = math.sqrt(float(df.iloc[i, header_pos])**2 +
                                float(df.iloc[i, header_pos+1])**2 +
                                float(df.iloc[i, header_pos+2])**2)
        if current_value > max_value:
            max_value = current_value  # Update max_value
            max_position = i  # Update max_position with the current row index

    print(f"The peak GRF is {max_value/bw} N/BW")
    print(f"The frame of peak GRF is {max_position}")

    header_name = User_Name + Sel_Part + "KneeAngles"
    page_start, page_end = page_positions['Model Outputs']
    header_pos = find_header_in_row(df, header_name, page_start)
    max_value = 0
    max_pos = 0
    
    for i in range(page_start + 3 + thres_frame, page_end):
        current_value = float(df.iloc[i, header_pos])
        if current_value <0:
            if (180 - abs(current_value)) > max_value:
                max_value = 180 - abs(current_value)
                max_pos = i   
        else:     
            if current_value > max_value:
                max_value = current_value
                max_pos = i

    print(f"Maximum Knee Angle: {max_value} at position: {max_pos}")
    reference_frame = int(df.iloc[max_pos, 0])
    print(reference_frame)
    print(df.iloc[max_pos, header_pos])

    # GRF value
    page_start, page_end = page_positions['Devices']
    header_pos = find_header_in_row(df, name_plates, page_start)
    GRF_y_pos = 0
    GRF_z_pos = 0
    GRF_y_cop = 0
    for i in range(10):  # Assuming the need to iterate over 8 rows as per the provided code
        GRF_y_pos += float(df.iloc[page_start + 3 + (reference_frame - 1) * 10 + i, header_pos + 1])
        GRF_z_pos += float(df.iloc[page_start + 3 + (reference_frame - 1) * 10 + i, header_pos + 2])
        GRF_y_cop += float(df.iloc[page_start + 3 + (reference_frame - 1) * 10 + i, header_pos + 7])
    GRF_y_cop /= 10
    GRF_YZ_cop = np.array([GRF_y_cop, 0])
    GRF_YZ_plane = np.array([GRF_y_pos / 10 + GRF_y_cop, -GRF_z_pos / 10])
    def extract_vector_single(name_header, name_page, Plane_Sel):
        header_name = User_Name + Sel_Part + name_header
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
    KJC_value = np.array(extract_vector_single("KJC","Model Outputs", "YZ")) 
    AJC_value = np.array(extract_vector_single("AJC","Model Outputs", "YZ"))
    ASIS_value = np.array(extract_vector_single("ASI", "Trajectories","YZ"))
    print(KJC_value,AJC_value,ASIS_value)
    FPKPA_value = calculate_angle_between_vectors(KJC_value- AJC_value, ASIS_value - KJC_value)
    print(f"FPKPA value is {FPKPA_value} degree")
    #GRF vector knee
    LFC_value = np.array(extract_vector_single("KNE", "Trajectories","YZ"))
    print(LFC_value)
    KJC_copy = KJC_value
    KJC_copy = KJC_value.copy()
    KJC_copy[1] = 0
    KJC_per = KJC_copy
    print(GRF_YZ_plane)
    KJC_LFC_angle = calculate_angle_between_vectors(KJC_value - KJC_per, LFC_value - KJC_per)
    GRF_KJC_angle = calculate_angle_between_vectors(KJC_value - KJC_per, GRF_YZ_plane - GRF_YZ_cop)
    GRF_LFC_angle = calculate_angle_between_vectors(LFC_value - KJC_per, GRF_YZ_plane - GRF_YZ_cop)

    check_GRF = np.array([GRF_YZ_plane [0] + GRF_YZ_cop[0], GRF_YZ_plane [1]])
    direction_LFC = check_left_or_right(KJC_value, LFC_value)
    direction_GRF = check_left_or_right(KJC_value,  check_GRF )

    print("LFC_value  is to the " + direction_LFC + " of KJC_value")
    print("GRF_YZ_plane" + direction_GRF + " of KJC_value")
    point = 1  # Default score if none of the other conditions are met

    if Sel_Part == 'R':
        if GRF_KJC_angle < GRF_LFC_angle and direction_GRF == 'right':
            point = 0
        elif GRF_KJC_angle > GRF_LFC_angle and direction_GRF == 'left':
            point = 2

    if Sel_Part == 'L':
        if GRF_KJC_angle < GRF_LFC_angle and direction_GRF == 'left':
            point = 0
        elif GRF_KJC_angle > GRF_LFC_angle and direction_GRF == 'right':
            point = 2
    #Pelvis Angle
    # Create a plot
    fig, ax = plt.subplots()

    # Plot LFC & KJC points
    ax.plot(LFC_value[0], LFC_value[1], 'bo')  # 'bo' creates a blue circle marker
    ax.text(LFC_value[0], LFC_value[1], 'LFC', verticalalignment='bottom')
    ax.plot(KJC_value[0], KJC_value[1], 'ro')  # 'ro' creates a red circle marker
    ax.text(KJC_value[0], KJC_value[1], 'KJC', verticalalignment='top')

    #draw GRF vector
    start = GRF_YZ_cop
    vector = GRF_YZ_plane
    ax.quiver(GRF_YZ_cop[0], GRF_YZ_cop[1], GRF_YZ_plane[0], GRF_YZ_plane [1], angles='xy', scale_units='xy', scale=1, color='k')

    plt.title(f'Leg: {Sel_Part} / Front view ') #limb side

    # Set the aspect of the plot to be equal, so circles look like circles,
    # and set a grid and some limits for clarity
    ax.set_aspect('equal')
    plt.grid(True)
    plt.xlim(0, 1000)
    plt.ylim(0, 1500)

    # Show the plot
    plt.show()
    RASI_name = User_Name + "RASI"
    LASI_name = User_Name + "LASI"
    page_start, page_end = page_positions['Trajectories']
    RASI_header_pos = find_header_in_row(df, RASI_name, page_start)
    LASI_header_pos = find_header_in_row(df, LASI_name, page_start)

    RASI_value = np.array([float(df.iloc[page_start + reference_frame + 2, RASI_header_pos + 1]),
                float(df.iloc[page_start + reference_frame + 2, RASI_header_pos + 2])])
    LASI_value = np.array([float(df.iloc[page_start + reference_frame + 2, LASI_header_pos + 1]),
                float(df.iloc[page_start + reference_frame + 2, LASI_header_pos + 2])])
    pelvis_angle = calculate_angle_between_vectors(LASI_value -RASI_value, np.array([1,0]))
    print(f"The angle between ASIS to ASIS line and horizontal neutral reference is {pelvis_angle} degree")
    #Trunk angle
    print(RASI_value, LASI_value)
    midline_pelvis = (RASI_value + LASI_value)/2
    CLAV_name = User_Name + "CLAV"
    page_start, page_end = page_positions['Trajectories']
    CLAV_header_pos = find_header_in_row(df, CLAV_name, page_start)
    CLAV_value = np.array([float(df.iloc[page_start + reference_frame + 2, CLAV_header_pos + 1]),
                float(df.iloc[page_start + reference_frame + 2, CLAV_header_pos + 2])])

    print(CLAV_value)
    trunk_angle = calculate_angle_between_vectors(CLAV_value -midline_pelvis, np.array([0,1]))
    print(trunk_angle)    
    #Knee flexion Check done
    ANK_value = np.array(extract_vector_single("ANK", "Trajectories","XZ"))
    KNE_value = np.array(extract_vector_single("KNE", "Trajectories","XZ"))
    THI_value = np.array(extract_vector_single("THI", "Trajectories","XZ"))
    print(ANK_value,KNE_value,THI_value)
    knee_flexion_angle = calculate_angle_between_vectors(ANK_value- KNE_value, THI_value - KNE_value)
    print(knee_flexion_angle)
    #Hip-Trunk flexion Check done

    SJC_value =  np.array(extract_vector_single("SJC", "Model Outputs","XZ"))
    print(SJC_value)
    hip_trunk_flexion_angle = calculate_angle_between_vectors(THI_value-SJC_value, THI_value - KNE_value)
    print(hip_trunk_flexion_angle)
    # find the distance of single hop test
    page_start, page_end = page_positions['Trajectories']
    header_name_Sel = User_Name + Sel_Part + "HEE"
    header_pos_Sel = find_header_in_row(df, header_name_Sel, page_start)

    col_Sel = [header_pos_Sel, header_pos_Sel + 1, header_pos_Sel + 2]
    col_Sel_Z = [header_pos_Sel + 2]

    Sel_Trajectories = df.iloc[(page_start+3):, col_Sel].reset_index(drop=True)
    Sel_Trajectories_Z = df.iloc[(page_start+3):, col_Sel_Z].reset_index(drop=True)

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

    # Assuming the peaks are already sorted by their height, take the first peak only
    if len(peaks) > 0:
        first_peak = peaks[0]
        first_peak_value = peak_values[0]

        # Ensure there is a first peak and calculate the minimum value in the window after the first peak
        window_start = first_peak
        window_end = min(window_start + 100, len(sel_trajectories_Z_array))
        min_value_index = np.argmin(sel_trajectories_Z_array[window_start:window_end]) + window_start
        min_value = sel_trajectories_Z_array[min_value_index]

    if first_non_empty_index is not None and len(peaks) >= 0:
        # Calculate the distance
        point_start = sel_trajectories_array[first_non_empty_index][0]
        point_end = sel_trajectories_array[min_value_index][0]
        vector = point_end - point_start
        projection_length = np.dot(vector, [1, 0, 0])
        projection_length = int(projection_length)
        print(f"The distance of Triple Hop Test in {Sel_Part} leg is: {projection_length} mm")
    else:
        print("Insufficient data to calculate the projection length.")



    scores = {}  # Dictionary to hold the scores
    scores['Distance'] = projection_length

    # Limb stability
    if FPKPA_value > 25:
        scores['FPKPA'] = 0
    elif FPKPA_value < 10:
        scores['FPKPA'] = 2
    else:
        scores['FPKPA'] = 1

    # GRF vs knee position
    scores['GRF and KJC'] = point

    # Pelvis stability
    if pelvis_angle > 10:
        scores['Pelvis Stability'] = 0
    elif pelvis_angle < 5:
        scores['Pelvis Stability'] = 2
    else:
        scores['Pelvis Stability'] = 1

    # Trunk stability
    if trunk_angle > 10:
        scores['Trunk Stability'] = 0
    elif trunk_angle < 5:
        scores['Trunk Stability'] = 2
    else:
        scores['Trunk Stability'] = 1

    # Shock absorption
    if knee_flexion_angle > 120:
        scores['Shock Absorption'] = 0
    elif knee_flexion_angle < 100:
        scores['Shock Absorption'] = 2
    else:
        scores['Shock Absorption'] = 1

    # Movement strategy
    movement_strategy_score = 2
    if knee_flexion_angle > 110:
        movement_strategy_score -= 1
    if hip_trunk_flexion_angle > 100:
        movement_strategy_score -= 1
    scores['Movement Strategy'] = movement_strategy_score

    return scores
    
     






