import pandas as pd
import math
import numpy as np

from header_work import find_header_in_row,  calculate_angle_between_vectors, check_left_or_right

def less_score(df, User_Name, page_positions, Left_Plate, Right_Plate, Kinetic_Sel, Thres_Hold_Init_Cont):
    """
    Finds the initial contact frames for left and right feet based on the given thresholds.

    Parameters:
    - df: DataFrame containing the force data.
    - page_positions: Dictionary with page start and end positions.
    - Left_Plate: Identifier for the left plate.
    - Right_Plate: Identifier for the right plate.
    - Kinetic_Sel: Kinetic selection identifier.
    - Thres_Hold_Init_Cont: Threshold for determining initial contact.

    Returns:
    - Tuple containing initial contact frame for left and right feet and the selected part.
    """
    Name_left_Plate = 'Imported AMTI 400600     #'+ Left_Plate+ ' - ' + Kinetic_Sel
    name_right_plate = 'Imported AMTI 400600     #'+ Right_Plate+ ' - ' + Kinetic_Sel
    print(name_right_plate)
    page_start, page_end = page_positions['Devices']
    header_pos_leftGRF = find_header_in_row(df, Name_left_Plate, page_start)
    header_pos_rightGRF = find_header_in_row(df, name_right_plate, page_start)

    for i in range(page_start+3, page_end):
      current_value_left = math.sqrt(float(df.iloc[i, header_pos_leftGRF])**2 +
                              float(df.iloc[i, header_pos_leftGRF+1])**2 +
                              float(df.iloc[i, header_pos_leftGRF+2])**2)
      if current_value_left > Thres_Hold_Init_Cont:
        print(f"Initial Contact of left foot: {current_value_left} at index {i}")
        initial_index_left = i
        thres_frame_left = int(df.iloc[i,0])
        print(f"Initial contact of left foot at frame: {thres_frame_left}")
        break

    for i in range(page_start+3, page_end):
      current_value_right = math.sqrt(float(df.iloc[i, header_pos_rightGRF])**2 +
                              float(df.iloc[i, header_pos_rightGRF+1])**2 +
                              float(df.iloc[i, header_pos_rightGRF+2])**2)
      if current_value_right > Thres_Hold_Init_Cont:
        print(f"Initial Contact of right foot: {current_value_left} at index {i}")
        initial_index_right = i
        thres_frame_right = int(df.iloc[i,0])
        print(f"Initial contact of right foot at frame: {thres_frame_right}")
        break

    print(f"Initial contact of left foot at frame: {thres_frame_left}")
    print(f"Initial contact of right foot at frame: {thres_frame_right}")
    print (initial_index_left)

    if thres_frame_left < thres_frame_right:
      sel_part = "L"
      frame_offset = thres_frame_right - thres_frame_left
      print(f"Left foot contact ground before right foot {thres_frame_right - thres_frame_left} frame")
    elif thres_frame_right == thres_frame_left:
      sel_part = "L"
      print(f"2 foot contact ground at the same time at frame {thres_frame_left}")
    else:
      frame_offset = thres_frame_left - thres_frame_right
      sel_part = "R"
      print(f"Right foot contact ground before left foot {thres_frame_left - thres_frame_right} frame")

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

    #Find maximum knee flexion angle & frame
    header_name = User_Name  + "RKneeAngles"
    page_start, page_end = page_positions['Model Outputs']
    header_pos = find_header_in_row(df, header_name, page_start)
    max_knee_angle = 0
    max_pos = 0
    for i in range(page_start+3+ thres_frame_right, page_end):
      current_value = df.iloc[i, header_pos]
      current_value = pd.to_numeric(current_value, errors='coerce')
      if current_value > max_knee_angle:
        max_knee_angle = current_value
        max_pos = i
    print(f"Maximum Knee Angle: {max_knee_angle} at position: {max_pos}")
    max_knee_reference_frame = int(df.iloc[max_pos,0])
    print(max_knee_reference_frame)

    #extract vectors in different 2D plane
    def extract_vector(name_header, name_page, Plane_Sel,sel_part,reference_frame):
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
        
    def check_left_or_right(reference_vec, other_vec):
        # This function assumes both vectors are in the YZ plane and treats them as 2D vectors.
        # Calculate a pseudo-cross product (just the determinant part since we're in 2D)
        cross_product_z = reference_vec[0] * other_vec[1] - reference_vec[1] * other_vec[0]

        if cross_product_z > 0:
            return "left"
        elif cross_product_z < 0:
            return "right"
        else:
            return "aligned"
        
    print(f"sel_part is {sel_part}")
    if sel_part == 'R':
      reference_frame = thres_frame_right
    else:
      reference_frame = thres_frame_left


    #Find contact frame & knee angle at it (POINT A)
    header_name = User_Name + sel_part + "KneeAngles"
    page_start, page_end = page_positions['Model Outputs']
    header_pos = find_header_in_row(df, header_name, page_start)


    if sel_part == "R":
      knee_flexion_initial = df.iloc[page_start + thres_frame_right +3,header_pos]
      print(f"The initial knee flexion angle of right foot is : {knee_flexion_initial}")
    else:
      knee_flexion_initial = df.iloc[page_start + thres_frame_left +3,header_pos]
      print(f"The initial knee flexion angle of left foot is : {knee_flexion_initial}")

    knee_flexion_initial = pd.to_numeric(knee_flexion_initial, errors='coerce')

    if knee_flexion_initial > 30:
      point_a = 0
    else:
      point_a = 1

    print (f"Point A = {point_a}")

    ### POINT B knee valgus at initital contact: check angle between: KJC to per_KJC and KJC to HEE + check left/right

    KJC_value = np.array(extract_vector("KJC","Model Outputs","YZ",sel_part,reference_frame))
    KJC_copy = KJC_value
    KJC_copy = KJC_value.copy()
    KJC_copy[1] = 0
    KJC_per = KJC_copy
    HEE_value = np.array(extract_vector("HEE","Trajectories","YZ",sel_part, reference_frame))
    KJC_HEE_angle = calculate_angle_between_vectors(KJC_value - KJC_per, HEE_value - KJC_per)
    direction_HEE = check_left_or_right(KJC_value - KJC_per, HEE_value - KJC_per)
    print(KJC_HEE_angle)
    print("HEE_value - KJC_per is to the " + direction_HEE + " of KJC_value - KJC_per")
    if sel_part == "R":
      if direction_HEE == "right" and KJC_HEE_angle > 5:
        point_b = 0
      else:
        point_b = 1
    else:
      if direction_HEE == "left" and KJC_HEE_angle > 5:
        point_b = 0
      else:
        point_b = 1  

    print(point_b)
    print (f"Point B = {point_b}")

    # # POINT C Trunk flexion at initial contact: Hip_Trunk angle in Single Leg Hop Score
    SJC_value =  np.array(extract_vector("SJC", "Model Outputs","XZ",sel_part, reference_frame))
    KNE_value = np.array(extract_vector("KNE", "Trajectories","XZ",sel_part, reference_frame))
    THI_value = np.array(extract_vector("THI", "Trajectories","XZ",sel_part, reference_frame))
    print(SJC_value)
    hip_trunk_flexion_angle = calculate_angle_between_vectors(THI_value-SJC_value, THI_value - KNE_value)
    print(hip_trunk_flexion_angle)
    if hip_trunk_flexion_angle > 100:
      point_c = 1
    else:
      point_c = 0
    print (f"Point C = {point_c}")

    ### POINT D Lateral Trunk Flexion: vertical when lateral trunk flexion angle < 5 degree
    RASI_name = User_Name + "RASI"
    LASI_name = User_Name + "LASI"
    page_start, page_end = page_positions['Trajectories']
    RASI_header_pos = find_header_in_row(df, RASI_name, page_start)
    LASI_header_pos = find_header_in_row(df, LASI_name, page_start)

    RASI_value = np.array([float(df.iloc[page_start + reference_frame + 2, RASI_header_pos + 1]),
                float(df.iloc[page_start + reference_frame + 2, RASI_header_pos + 2])])
    LASI_value = np.array([float(df.iloc[page_start + reference_frame + 2, LASI_header_pos + 1]),
                float(df.iloc[page_start + reference_frame + 2, LASI_header_pos + 2])])
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
    print(f"Lateral trunk flexion at contact is {trunk_angle} degree")
    if trunk_angle < 3:
      point_d = 0
    else: 
      point_d = 1
    print(f"Point D = {point_d}")


    # POINT E Ankle plantar flexion at contact: Check AnkleAngles_x, if < 0 -> point E = 0
    header_name = User_Name + sel_part + "AnkleAngles"
    page_start, page_end = page_positions['Model Outputs']
    ankle_header_pos = find_header_in_row(df, header_name, page_start)
    ankle_plantar_angle = float(df.iloc[page_start + reference_frame + 2, ankle_header_pos + 1])
    print(ankle_plantar_angle)
    if ankle_plantar_angle < 0:
      point_e = 0
    else:
      point_e = 1
    print(f"Point E = {point_e}")


    #POINT F & G: vẽ line HEE và TOE (XY plane), tính angle của line đó và trục x (1,0)
    #F: >30 =1, G: <-30: =1 else = 0
    HEE_value = np.array(extract_vector("HEE","Trajectories","XY",sel_part, reference_frame))
    TOE_value = np.array(extract_vector("TOE","Trajectories","XY",sel_part, reference_frame))
    angle_rotation = calculate_angle_between_vectors(HEE_value - TOE_value, [1,0])
    print(angle_rotation)
    if angle_rotation > 30:
      point_f = 1
    else:
      point_f = 0
    if angle_rotation < -30:
      point_g = 1
    else:
      point_g = 0
    print(point_f)
    print(point_g)
    print(f"Point F = {point_f}")
    print(f"Point G = {point_g}")


    ### POINT H & I : stance width vs shoulder width
    page_start, page_end = page_positions['Trajectories']
    LHEE_header_pos = find_header_in_row(df, 'LHEE', page_start)
    RHEE_header_pos = find_header_in_row(df, 'RHEE', page_start)
    LHEE_trajectory_contact = np.array([float(df.iloc[page_start + reference_frame + 2, LHEE_header_pos]), float(df.iloc[page_start + reference_frame + 2, LHEE_header_pos + 1]), float(df.iloc[page_start + reference_frame + 2, LHEE_header_pos + 2])])
    print(LHEE_trajectory_contact)
    RHEE_trajectory_contact = np.array([float(df.iloc[page_start + reference_frame + 2, RHEE_header_pos]), float(df.iloc[page_start + reference_frame + 2, RHEE_header_pos + 1]), float(df.iloc[page_start + reference_frame + 2, RHEE_header_pos + 2])])
    print(LHEE_trajectory_contact)
    stance_width = np.linalg.norm(RHEE_trajectory_contact - LHEE_trajectory_contact)
    print(stance_width)

    LSHO_header_pos = find_header_in_row(df, 'LSHO', page_start)
    RSHO_header_pos = find_header_in_row(df, 'RSHO', page_start)
    LSHO_trajectory_contact = np.array([float(df.iloc[page_start + reference_frame + 2, LSHO_header_pos]), float(df.iloc[page_start + reference_frame + 2, LSHO_header_pos + 1]), float(df.iloc[page_start + reference_frame + 2, LSHO_header_pos + 2])])
    print(LSHO_trajectory_contact)
    RSHO_trajectory_contact = np.array([float(df.iloc[page_start + reference_frame + 2, RSHO_header_pos]), float(df.iloc[page_start + reference_frame + 2, RSHO_header_pos + 1]), float(df.iloc[page_start + reference_frame + 2, RSHO_header_pos + 2])])
    print(LSHO_trajectory_contact)
    shoulder_width = np.linalg.norm(RSHO_trajectory_contact - LSHO_trajectory_contact)
    print(shoulder_width)
    point_h = 0
    point_i = 0
    if stance_width < shoulder_width - 20:
      point_h = 1
    elif shoulder_width < stance_width - 20:
      point_i = 1
    else:
      point_h  = 0
      point_i = 0
    print(point_h)
    print(point_i)
    print(f"Point H = {point_h}")
    print(f"Point I = {point_i}")


    #### POINT J: Initial foot contact
    if thres_frame_left == thres_frame_right:
      point_j = 0
    else:
      point_j = 1
    print(f"Point J = {point_j}")


    # POINT K : Knee flexion before jumping
    page_start, page_end = page_positions['Trajectories']
    LHEE_header_pos = find_header_in_row(df, 'LHEE', page_start)
    RHEE_header_pos = find_header_in_row(df, 'RHEE', page_start)

    RHEE_trajectory_before_Z = df.iloc[page_start + 3, RHEE_header_pos + 2]
    RHEE_trajectory_before_Z = pd.to_numeric(RHEE_trajectory_before_Z, errors='coerce')

    LHEE_trajectory_before_Z = df.iloc[page_start + 2, LHEE_header_pos + 2]
    LHEE_trajectory_before_Z = pd.to_numeric(LHEE_trajectory_before_Z, errors='coerce')
    start_jumping_frame = 0
    start_jumping_pos = 0
    for i in range(page_start+3, page_end):
      current_value_heel_z = df.iloc[i, RHEE_header_pos + 2]
      current_value_heel_z = pd.to_numeric(current_value_heel_z, errors='coerce')
      if current_value_heel_z - RHEE_trajectory_before_Z > 15 :
        start_jumping_pos = i
        start_jumping_frame =  i - page_start - 2
        break
    print(start_jumping_pos)
    print(start_jumping_frame)

    header_name = User_Name  + "RKneeAngles"
    page_start, page_end = page_positions['Model Outputs']
    knee_angle_header_pos = find_header_in_row(df, header_name, page_start)
    knee_flex_before = df.iloc[page_start + 3 + start_jumping_frame, knee_angle_header_pos]
    knee_flex_before = pd.to_numeric(knee_flex_before, errors='coerce')
    print(f"Knee flexion before jumping is {knee_flex_before} degree")

    if knee_flex_before > 45:
      point_k = 0
    else:
      point_k = 1
    print(f"Point K = {point_k}")


    ######### Point L: 
    KNE_value = np.array(extract_vector("KNE","Trajectories","YZ",sel_part,start_jumping_frame))
    KNE_copy = KNE_value
    KNE_copy = KNE_value.copy()
    KNE_copy[1] = 0
    KNE_per = KNE_copy
    BTOE_value = np.array(extract_vector("BTOE","Trajectories","YZ",sel_part, start_jumping_frame))
    KNE_BTOE_angle = calculate_angle_between_vectors(KNE_value - KNE_per, BTOE_value - KNE_per)
    direction_BTOE = check_left_or_right(KNE_value - KNE_per, BTOE_value - KNE_per)
    print(KNE_BTOE_angle)
    print("BTOE_value - KNE_per is to the " + direction_BTOE + " of KNE_value - KNE_per")
    if sel_part == "R":
      if direction_BTOE == "right" and KNE_BTOE_angle > 2:
        point_l = 0
      else:
        point_l = 1
    else:
      if direction_BTOE == "left" and KNE_BTOE_angle > 2:
        point_l = 0
      else:
        point_l = 1

    print(point_l)
    print (f"Point L = {point_l}")


    #M.  Trunk flexion at maximal knee angle, trunk flexed more than at initial contact; 0  = yes, 1 = no
    SJC_value =  np.array(extract_vector("SJC", "Model Outputs","XZ",sel_part,max_knee_reference_frame))
    KNE_value = np.array(extract_vector("KNE", "Trajectories","XZ",sel_part,max_knee_reference_frame))
    THI_value = np.array(extract_vector("THI", "Trajectories","XZ",sel_part,max_knee_reference_frame))
    hip_trunk_angle_max_knee = calculate_angle_between_vectors(THI_value-SJC_value, THI_value - KNE_value)
    print(hip_trunk_angle_max_knee)
    print(hip_trunk_flexion_angle)
    if hip_trunk_angle_max_knee < hip_trunk_flexion_angle:
      point_m = 1
    else:
      point_m = 0
    print(point_m)
   
    print(f"Point M = {point_m}")


    #N.  Hip flexion angle at initial contact, hips flexed; 0 = yes, 1 = no
    Hip_angle_value =  np.array(extract_vector("HipAngles", "Model Outputs","X",sel_part,reference_frame))
    print(Hip_angle_value)
    if Hip_angle_value > 30:
      point_n = 0
    else:
      point_n = 1
    # Set the threshold for flexed or not
    print(f"Point N = {point_n}")


    #O  Hip flexion at maximal knee angle, hips flexed more than at initial contact; 0 = yes, 1  = no
    Hip_angle_value_max_knee =  np.array(extract_vector("HipAngles", "Model Outputs","X",sel_part,max_knee_reference_frame))
    print(Hip_angle_value_max_knee)
    if Hip_angle_value_max_knee > Hip_angle_value:
      point_o = 0
    else:
      point_o = 1
    print(point_o)
    print(f"Point O = {point_o}")


    print ("--------------------------------------------------------------------------------------")
    
  # Assign scores for each point
    score = {}
    score['Point A'] = point_a
    score['Point B'] = point_b
    score['Point C'] = point_c
    score['Point D'] = point_d
    score['Point E'] = point_e
    score['Point F'] = point_f
    score['Point G'] = point_g
    score['Point H'] = point_h
    score['Point I'] = point_i
    score['Point J'] = point_j
    score['Point K'] = point_k
    score['Point L'] = point_l
    score['Point M'] = point_m
    score['Point N'] = point_n
    score['Point O'] = point_o


    return score


