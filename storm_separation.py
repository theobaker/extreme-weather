import os
import gzip
import pandas as pd
from datetime import datetime, timedelta
import re

# Directory where the downloaded files are stored
input_dir = "/Users/theobaker/Downloads/109final/"
output_file = "/Users/theobaker/Downloads/109final/2004-2018_election_day_extreme_events.csv"

# Define extreme weather event types
EXTREME_EVENTS = [
    "Tornado", "Hurricane (Typhoon)", "Flood", "Flash Flood", 
    "Winter Storm", "Ice Storm", "Blizzard", "Excessive Heat", 
    "Extreme Cold/Wind Chill", "Storm Surge/Tide", "Wildfire"
]

# Calculate election days for each even year, including off-cycle elections
def get_election_days(start_year, end_year):
    election_days = {}
    for year in range(start_year, end_year + 1):  # All years in the range
        if year % 2 == 0:  # Consider only even years
            nov_1st = datetime(year, 11, 1)
            # Find the first Tuesday after the first Monday
            first_tuesday = nov_1st + timedelta(days=(1 - nov_1st.weekday() + 7) % 7)
            election_days[year] = first_tuesday.date()
    return election_days


def process_files(input_dir, election_days):
    results = []

    for file_name in os.listdir(input_dir):
        # Skip non-gzip files
        if not file_name.endswith(".gz"):
            continue

        # Use regex to extract the year from the filename
        match = re.search(r"_d(\d{4})", file_name)
        if not match:
            print(f"Skipping file due to unexpected format: {file_name}")
            continue
        year = int(match.group(1))

        # Skip odd years or non-matching years
        if year not in election_days:
            continue

        # Process only "details" files
        if "details" not in file_name:
            print(f"Skipping non-details file: {file_name}")
            continue

        # Read the gzip file
        file_path = os.path.join(input_dir, file_name)
        try:
            with gzip.open(file_path, 'rt') as f:
                data = pd.read_csv(f)
        except (gzip.BadGzipFile, OSError) as e:
            print(f"Error reading gzip file {file_name}: {e}")
            continue

        # Check if 'BEGIN_DATE_TIME' exists
        if 'BEGIN_DATE_TIME' not in data.columns:
            print(f"Skipping file {file_name} as 'BEGIN_DATE_TIME' is not present.")
            continue

        # Parse 'BEGIN_DATE_TIME' with explicit format
        try:
            data['BEGIN_DATE'] = pd.to_datetime(data['BEGIN_DATE_TIME'], format='%d-%b-%y %H:%M:%S', errors='coerce')
        except Exception as e:
            print(f"Date parsing failed for file {file_name}: {e}")
            continue

        # Drop rows with invalid dates
        data = data[data['BEGIN_DATE'].notna()]
        data['DATE_ONLY'] = data['BEGIN_DATE'].dt.date

        # Filter for election days and extreme events
        election_date = election_days.get(year)
        date_range = [election_date - timedelta(days=1), election_date, election_date + timedelta(days=1)]
        filtered = data[
         (data['DATE_ONLY'].isin(date_range)) & 
         (data['EVENT_TYPE'].str.strip().isin(EXTREME_EVENTS))
        ]

        print(f"Filtered events in {file_name}: {len(filtered)} rows")
        if not filtered.empty:
            results.append(filtered)

        # Debugging Election Day and Filter Results
        print(f"Election day for {year}: {election_days.get(year)}")
        unique_dates = data['DATE_ONLY'].unique()
        print(f"Available dates in data: {unique_dates}")

        if election_days.get(year) not in unique_dates:
          print(f"No events occurred on election day {election_days.get(year)} for {year}.")
        else:
          print(f"Events occurred on election day {election_days.get(year)} for {year}.")


    # Combine all results into a single DataFrame
    if results:
        combined = pd.concat(results, ignore_index=True)
        print(f"Total filtered events: {len(combined)}")
        return combined
    else:
        print("No events matched the criteria.")
        return pd.DataFrame()

# Main function
def main():
    # Define the range of years to process
    start_year = 2004
    end_year = 2018

    # Calculate election days
    election_days = get_election_days(start_year, end_year)

    # Process files and extract relevant events
    extreme_events = process_files(input_dir, election_days)

    # Save results to a CSV file
    if not extreme_events.empty:
      extreme_events = extreme_events.sort_values(by='BEGIN_DATE')  # Sort by date
      extreme_events.to_csv(output_file, index=False)
      print(f"Filtered extreme weather events saved to {output_file}")
    else:
     print("No extreme weather events matched the criteria.")


# Run the script
if __name__ == "__main__":
    main()