import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file
file_path = "../../../Notebooks/Timo-Eindhoven-Arnhem.csv"
file_path_Eindhoven_Arhem = "../../../Notebooks/Timo-Eindhoven-Arnhem.csv"
file_path_Eindhoven_DenBosch = "../../../Notebooks/Timo-Eindhoven-DenBosch.csv"
file_path_Maarheeze_Eindhoven = "../../../Notebooks/Timo-Maarheeze-Eindhoven.csv"

df_Eindhoven_Arhem = pd.read_csv(file_path_Eindhoven_Arhem)
df_Eindhoven_DenBosch = pd.read_csv(file_path_Eindhoven_DenBosch)
df_Maarheeze_Eindhoven = pd.read_csv(file_path_Maarheeze_Eindhoven)

df_Maarheeze_Arhem = pd.concat([df_Maarheeze_Eindhoven, df_Eindhoven_Arhem], ignore_index=True)
df_Maarheeze_DenBosch = pd.concat([df_Maarheeze_Eindhoven, df_Eindhoven_DenBosch], ignore_index= True)

df = df_Maarheeze_Arhem
ndw_df = pd.read_pickle("../../../Notebooks/DONE_24h_intensity-speed-Maarheeze-to-Arnhem_01-12-22_30-11-23.pkl")

# df = df_Maarheeze_DenBosch
# ndw_df = pd.read_pickle("../../../Notebooks/DONE_24h_intensity-speed-Maarheeze-to-sHertogenbosch_01-12-22_30-11-23")

# List of columns to keep
columns_to_keep = [
    "File Start Date", "File End Date", "File Start Time", "File End Time", 
    "File Duration", "Hectometer Head", "Hectometer Tail", "Route Letter", 
    "Route Number", "Route Description", "Hectometering Direction", "Trajectory From", "Trajectory To", "Route"
]

# Drop all other columns except the ones listed above
df_filtered = df[columns_to_keep]

# Display the first few rows of the filtered dataframe
df_filtered.info()

# Create a copy of the filtered DataFrame to avoid SettingWithCopyWarning
df_filtered_copy = df_filtered.copy()

# Converting time columns to datetime for easier filtering
df_filtered_copy['File Start Time'] = pd.to_datetime(df_filtered_copy['File Start Time'], format='%H:%M:%S').dt.time
df_filtered_copy['File End Time'] = pd.to_datetime(df_filtered_copy['File End Time'], format='%H:%M:%S').dt.time

# Adding a new column to check if the start hour is different from the end hour
df_filtered_copy['Different Start-End Hour'] = df_filtered_copy['File Start Time'].apply(lambda x: x.hour) != df_filtered_copy['File End Time'].apply(lambda x: x.hour)

df_filtered_copy.info()




from datetime import datetime, timedelta
import numpy as np

def calculate_absolute_hectometer_per_minute(row):
    """
    Calculate the absolute value of hectometers per minute for a given row.
    """
    start_datetime = datetime.strptime(f"{row['File Start Date']} {row['File Start Time']}", '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(f"{row['File End Date']} {row['File End Time']}", '%Y-%m-%d %H:%M:%S')
    total_minutes = (end_datetime - start_datetime).total_seconds() / 60
    hectometer_distance = abs(row['Hectometer Tail'] - row['Hectometer Head'])
    return hectometer_distance / total_minutes if total_minutes > 0 else 0

def split_rows_with_absolute_hpm(data):
    """
    Split rows based on different hours in 'File Start Time' and 'File End Time'
    and also consider different dates. The split will end at XX:59:59. 
    'Hectometer per Minute' is calculated as an absolute value.
    """
    new_data = []
    for index, row in data.iterrows():
        start_date_str = str(row['File Start Date'])
        end_date_str = str(row['File End Date'])
        start_time_str = str(row['File Start Time'])
        end_time_str = str(row['File End Time'])

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        start_time = datetime.strptime(start_time_str, '%H:%M:%S')
        end_time = datetime.strptime(end_time_str, '%H:%M:%S')

        hpm = calculate_absolute_hectometer_per_minute(row)

        start_dt = datetime.combine(start_date, start_time.time())
        end_dt = datetime.combine(end_date, end_time.time())

        while start_dt < end_dt:
            new_row = row.copy()
            new_row['File Start Date'] = start_dt.strftime('%Y-%m-%d')
            new_row['File Start Time'] = start_dt.strftime('%H:%M:%S')

            next_hour = (start_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)) - timedelta(seconds=1)
            segment_end_dt = min(next_hour, end_dt)

            new_row['File End Date'] = segment_end_dt.strftime('%Y-%m-%d')
            new_row['File End Time'] = segment_end_dt.strftime('%H:%M:%S')

            time_span = (segment_end_dt - start_dt).total_seconds() / 60
            hectometer_span = time_span * hpm
            new_row['Hectometer Tail'] = row['Hectometer Head'] + hectometer_span
            new_row['Hectometer per Minute'] = hpm

            new_data.append(new_row)

            start_dt = segment_end_dt + timedelta(seconds=1)
            row['Hectometer Head'] = new_row['Hectometer Tail']

    return pd.DataFrame(new_data)

# Applying the function with absolute hectometer per minute to the dataset
split_data_with_absolute_hpm = split_rows_with_absolute_hpm(df_filtered_copy)  # Testing with a smaller subset

data_df = split_data_with_absolute_hpm
# Adding sorting by 'File Duration' in descending order at the end of the process


data_df.info()



# Convert 'File Start Time' and 'File End Time' to datetime
data_df['File Start Time'] = pd.to_datetime(data_df['File Start Time'], format='%H:%M:%S')
data_df['File End Time'] = pd.to_datetime(data_df['File End Time'], format='%H:%M:%S')

# Adding a dummy date to the time fields
dummy_date = datetime(2000, 1, 1)  # The date doesn't matter, it's just a placeholder

# Convert time to string and concatenate with dummy date
data_df['start_datetime_full'] = pd.to_datetime(
    dummy_date.strftime('%Y-%m-%d') + ' ' + data_df['File Start Time'].dt.strftime('%H:%M:%S')
)
data_df['end_datetime_full'] = pd.to_datetime(
    dummy_date.strftime('%Y-%m-%d') + ' ' + data_df['File End Time'].dt.strftime('%H:%M:%S')
)

# Calculate the time difference
data_df['time_difference'] = data_df['end_datetime_full'] - data_df['start_datetime_full']

# Convert time difference to total seconds
data_df['time_difference_seconds'] = data_df['time_difference'].dt.total_seconds()

# Convert time difference to total minutes
data_df['time_difference_minutes'] = data_df['time_difference_seconds'] / 60

# Remove the dummy date, keeping only the time
data_df['start_time'] = data_df['start_datetime_full'].dt.time
data_df['end_time'] = data_df['end_datetime_full'].dt.time

# Display the DataFrame
data_df.head()


data_df.to_csv('forGPT.csv', index=False)


# Function to generate time ranges (e.g., 07:00 - 08:00)
def generate_time_ranges(start_hour, end_hour):
    ranges = []
    for hour in range(start_hour, end_hour):
        start_time = datetime.strptime(f"{hour:02d}:00", "%H:%M")
        end_time = start_time + timedelta(hours=1)
        ranges.append((start_time.time(), end_time.time()))
    return ranges
 
# Generate time ranges from 00:00 to 23:00
time_ranges = generate_time_ranges(0, 24)
 
# Function to assign a time range to a datetime
def assign_time_range(dt, time_ranges):
    for start, end in time_ranges:
        if start <= dt.time() < end:
            return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
    return None
 
# Assigning time ranges to each record
data_df['Time Period'] = data_df['File Start Time'].apply(lambda dt: assign_time_range(dt, time_ranges))

# Calculate the delay per hectometer
data_df['HM Difference'] = data_df['Hectometer Tail'] - data_df['Hectometer Head']
data_df['Time Difference'] = (data_df['File End Time'] - data_df['File Start Time']).dt.total_seconds() / 60  # in minutes
data_df['Min. per HM'] = data_df['Time Difference'] / data_df['HM Difference']
 
# Function to split the data for every 0.1 HM increment
def split_hm_sections(row):
    hm_start = row['Hectometer Head']
    hm_end = row['Hectometer Tail']
    hm_sections = []
 
    # Generate 0.1 HM increments within the range
    while hm_start < hm_end:
        next_hm = min(hm_start + 0.1, hm_end)
        hm_sections.append({
            'Date': row['File Start Date'],
            'Time Period': row['Time Period'],
            'HM Section': round(hm_start, 1),
            'Min. per HM': row['Min. per HM'] * (next_hm - hm_start)
        })
        hm_start = next_hm
 
    return hm_sections
 


# Apply the function to each row and create a new DataFrame
split_data = pd.DataFrame([item for _, row in data_df.iterrows() for item in split_hm_sections(row)])

split_data.info()
split_data.to_csv("../Merge/Maarheeze_Arhem.csv")




# Grouping the data by 'Time Period' and 'File Start Date' and summing the 'Time Difference Minutes'
grouped_data = data_df.groupby(['Time Period', 'File Start Date'])['time_difference_minutes'].sum().reset_index()

#Renaming the column for clarity
grouped_data.rename(columns={'time_difference_minutes': 'Total Time Difference (Minutes)'}, inplace=True)

grouped_data.to_csv('nonTimeSort.csv', index=False)
#grouped_data.info



import re
grouped_data = split_data

# Function to convert time period and date into a single datetime object
def convert_to_datetime(date_str, time_period):
    if time_period is None:
        return None  # or some default value e.g., datetime.min
    start_hour = int(time_period.split(':')[0])
    datetime_str = f"{date_str} {start_hour:02d}:00"
    return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')

# Applying the function to each row
grouped_data['Datetime'] = grouped_data.apply(lambda row: convert_to_datetime(row['Date'], row['Time Period']), axis=1)

# Sorting the data by the new datetime column
sorted_data = grouped_data.sort_values(by='Datetime')

sorted_data.to_csv('TimeSort.csv', index=False)

# Displaying the sorted data
sorted_data.head(1000)


# Taking the first part of the "Time Period" and just the hour part
sorted_data['Hour'] = sorted_data['Time Period'].str.split('-').str[0].str.strip().str[:2]
    
# Read the Weather data
weather_df = pd.read_csv('../../../Notebooks/Timo-Weather.csv')
columns_to_drop = ['DD','FF','FX','P','VV','N','U','M','R','S','O','Y','Q','SQ','TD','DR']
weather_df = weather_df.drop(columns_to_drop, axis=1)
weather_df.rename(columns={
    'FH': 'Wind Speed',
    'T': 'Temperature',
    'RH': 'Rainfall'
}, inplace=True)
    
# Ensure that 'Hour' in weather and ndw data is a string for proper matching
weather_df['Hour'] = weather_df['Hour'].astype(str).str.zfill(2)  # Adding leading zero if needed
ndw_df['Hour'] = ndw_df['hour_of_day'].astype(str).str.zfill(2)  # Adding leading zero if needed

merged_df = pd.merge(sorted_data, weather_df, how='left', left_on=['Date', 'Hour'], right_on=['Date', 'Hour'])

# Convert 'start_measurement_period' to datetime objects
ndw_df['start_measurement_period'] = pd.to_datetime(ndw_df['start_measurement_period'])
ndw_df['Date'] = ndw_df['start_measurement_period'].dt.date.astype(str)
ndw_df.rename(columns={
    'average_intensity': 'Average Intensity',
    'average_speed': 'Average Speed'
}, inplace=True)

ndw_df['hectometer'] = ndw_df['name_measurement_location'].str.extract('thv hmp (\d+\.\d+)')
ndw_df['hectometer'] = ndw_df['hectometer'].astype(float)

# Function to find the closest hectometer
def find_closest_hectometer(target, hectometers):
    # Find the index of the closest hectometer
    closest_index = (hectometers - target).abs().argsort()[0]
    # Return the closest hectometer and the corresponding index
    return hectometers.iloc[closest_index], closest_index

# Apply the function and split the results into two new columns
ndw_df[['Closest Hectometer', 'Closest_Index']] = merged_df.apply(
    lambda row: pd.Series(find_closest_hectometer(row['HM Section'], ndw_df['hectometer'])),
    axis=1
)
columns_to_keep = ['Hour', 'Date', 'Average Intensity', 'Average Speed']
ndw_df = ndw_df[columns_to_keep]

merged_df = pd.merge(merged_df, ndw_df, how='left', left_on=['Date', 'Hour', 'HM Section'], right_on=['Date','Hour','Closest Hectometer'])

# RH, T, FH
ndw_df.head(1000)


merged_df.to_pickle("FINAL_Maarheeze_Arnhem.pkl")
# merged_df.to_pickle("FINAL_Maarheeze_Den_Bosch.pkl")