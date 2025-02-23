columns_less = {}

# Starting ASCII value for column 'B' (since 'A' is 65, 'B' is 66)
start_column = 66  # ASCII value for 'B'

# Generate column letters for each point from 'Point A' to 'Point O'
for i in range(15):  # There are 15 points from A to O
    point_label = f"Point {chr(ord('A') + i)}"  # Create label like 'Point A', 'Point B', etc.
    columns_less[point_label] = chr(start_column + i)  # Map to corresponding Excel column

# Print the dictionary to verify
print(columns_less)