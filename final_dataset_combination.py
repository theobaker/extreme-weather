import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

def load_and_clean_data(gazetteer_path, weather_path, voting_path):
    """Load and clean datasets"""
    # Load datasets
    gazetteer_df = pd.read_csv(gazetteer_path, delimiter="\t", dtype=str)
    weather_df = pd.read_csv(weather_path, dtype={
        'STATE_FIPS': str,
        'CZ_FIPS': str,
        'YEAR': str,
        'EVENT_ID': str,
        'EPISODE_ID': str
    })
    voting_df = pd.read_csv(voting_path, dtype={'stcofips': str, 'year': str})
    
    # Clean gazetteer data
    gazetteer_df.columns = gazetteer_df.columns.str.strip()
    gazetteer_df["GEOID"] = gazetteer_df["GEOID"].str.zfill(5)
    gazetteer_df["INTPTLAT"] = pd.to_numeric(gazetteer_df["INTPTLAT"], errors="coerce")
    gazetteer_df["INTPTLONG"] = pd.to_numeric(gazetteer_df["INTPTLONG"], errors="coerce")
    
    # Clean weather events data
    weather_df["STATE_FIPS"] = weather_df["STATE_FIPS"].str.zfill(2)
    weather_df["CZ_FIPS"] = weather_df["CZ_FIPS"].str.zfill(3)
    weather_df["FIPS"] = weather_df["STATE_FIPS"] + weather_df["CZ_FIPS"]
    
    # Convert coordinates and handle missing values
    for col in ['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON']:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
    
    # For events with missing BEGIN coordinates, use END coordinates if available
    weather_df.loc[weather_df['BEGIN_LAT'].isna(), 'BEGIN_LAT'] = weather_df['END_LAT']
    weather_df.loc[weather_df['BEGIN_LON'].isna(), 'BEGIN_LON'] = weather_df['END_LON']
    
    # Clean voting data
    voting_df["stcofips"] = voting_df["stcofips"].str.zfill(5)
    voting_df["year"] = voting_df["year"].astype(str)
    voting_df = voting_df[voting_df["year"].astype(int).between(2004, 2018)]
    
    return gazetteer_df, weather_df, voting_df

def get_zone_centroid(zone_name, state_fips, gazetteer_df):
    """Calculate the centroid of counties that might be part of a zone"""
    zone_name = zone_name.upper()
    
    # Create reference point mappings for common geographic features
    geographic_features = {
        'BLACK HILLS': {
            '46': (44.0, -103.8),  # SD Black Hills approximate center
            '56': (44.3, -104.1)   # WY Black Hills approximate center
        },
        'YAMPA RIVER': {
            '08': (40.5, -106.8)   # CO Yampa River Basin approximate center
        },
        'ROCKY MOUNTAIN FRONT': {
            '30': (47.3, -112.4)   # MT Rocky Mountain Front approximate center
        },
        'WET MOUNTAINS': {
            '08': (38.1, -105.2)   # CO Wet Mountains approximate center
        }
    }
    
    # Check for known geographic features
    for feature, state_coords in geographic_features.items():
        if feature in zone_name and state_fips in state_coords:
            return state_coords[state_fips]
    
    # Handle special Alaska regions
    if state_fips == '02':  # Alaska
        alaska_regions = {
            'KUSKOKWIM DELTA': (60.5, -162.3),
            'P.W. SND': (60.8, -147.5),  # Prince William Sound
            'NERN P.W.': (60.9, -146.5),  # Northern Prince William Sound
            'SERN P.W.': (60.3, -148.0)   # Southern Prince William Sound
        }
        for region, coords in alaska_regions.items():
            if region in zone_name:
                return coords
    
    # Try original county-based matching as fallback
    state_counties = gazetteer_df[gazetteer_df["GEOID"].str[:2] == state_fips]
    
    # Remove common geographic terms and qualifiers
    remove_terms = ['COUNTY', 'COUNTIES', 'INCLUDING', 'AND', '&', 'BETWEEN', 
                   'ABOVE', 'BELOW', 'FT', 'FEET', 'CENTRAL', 'EASTERN', 'WESTERN',
                   'NORTHERN', 'SOUTHERN', 'MOUNTAINS', 'RIVER', 'BASIN', 'VALLEY',
                   'FOOTHILLS', 'FRONT', 'RANGE']
    
    clean_name = zone_name
    for term in remove_terms:
        clean_name = clean_name.replace(term, '')
    
    # Split into potential location names
    keywords = [k.strip() for k in clean_name.split() if k.strip() and len(k.strip()) > 2]
    
    matching_counties = []
    for _, county in state_counties.iterrows():
        county_name = county["NAME"].upper()
        if any(keyword in county_name for keyword in keywords):
            matching_counties.append(county)
    
    if matching_counties:
        lats = [float(county["INTPTLAT"]) for county in matching_counties]
        lons = [float(county["INTPTLONG"]) for county in matching_counties]
        return np.mean(lats), np.mean(lons)
    
    # For cities (like "RAPID CITY"), try matching just the city name
    if 'CITY' in zone_name:
        city_name = zone_name.replace('CITY', '').strip()
        for _, county in state_counties.iterrows():
            if city_name in county["NAME"].upper():
                return float(county["INTPTLAT"]), float(county["INTPTLONG"])
    
    return None, None

def find_closest_counties(lat, lon, state_fips, gazetteer_df, k=5):
    """Find k closest counties using coordinates"""
    if pd.isna(lat) or pd.isna(lon):
        return None
        
    try:
        event_coords = np.radians([[lat, lon]])
        state_counties = gazetteer_df[gazetteer_df["GEOID"].str[:2] == state_fips]
        
        if len(state_counties) == 0:
            return None
            
        county_coords = np.radians(state_counties[["INTPTLAT", "INTPTLONG"]].values)
        distances = haversine_distances(event_coords, county_coords)[0] * 6371
        
        closest_indices = np.argsort(distances)[:k]
        
        return [(state_counties.iloc[i]["GEOID"], distances[i]) 
                for i in closest_indices]
    except (IndexError, KeyError):
        return None

def process_weather_events(weather_df, gazetteer_df):
    results = []
    stats = {'C': {'processed': 0, 'skipped': 0}, 
             'Z': {'processed': 0, 'skipped': 0}}
    
    for _, event in weather_df.iterrows():
        counties_with_distances = None
        
        # Handle county events
        if event['CZ_TYPE'] == 'C':
            # First try direct FIPS matching
            counties_with_distances = [(event['FIPS'], 0.0)]
            
            # Then find nearby counties using coordinates if available
            if pd.notna(event['BEGIN_LAT']) and pd.notna(event['BEGIN_LON']):
                nearby = find_closest_counties(
                    event['BEGIN_LAT'],
                    event['BEGIN_LON'],
                    event['STATE_FIPS'],
                    gazetteer_df,
                    k=4  # We already have the exact county
                )
                if nearby:
                    counties_with_distances.extend(nearby)
            
            if counties_with_distances:
                stats['C']['processed'] += 1
            else:
                stats['C']['skipped'] += 1
                continue
                
        # Handle zone events
        elif event['CZ_TYPE'] == 'Z':
            # First try using event coordinates
            if pd.notna(event['BEGIN_LAT']) and pd.notna(event['BEGIN_LON']):
                counties_with_distances = find_closest_counties(
                    event['BEGIN_LAT'],
                    event['BEGIN_LON'],
                    event['STATE_FIPS'],
                    gazetteer_df,
                    k=5
                )
            
            # If no coordinates, try to estimate from zone name
            if not counties_with_distances:
                lat, lon = get_zone_centroid(event['CZ_NAME'], event['STATE_FIPS'], gazetteer_df)
                if lat is not None and lon is not None:
                    counties_with_distances = find_closest_counties(
                        lat,
                        lon,
                        event['STATE_FIPS'],
                        gazetteer_df,
                        k=5
                    )
            
            if counties_with_distances:
                stats['Z']['processed'] += 1
            else:
                stats['Z']['skipped'] += 1
                continue
        
        # Add results for each county
        for county_fips, distance in counties_with_distances:
            try:
                county = gazetteer_df[gazetteer_df["GEOID"] == county_fips].iloc[0]
                results.append({
                    'EVENT_ID': event['EVENT_ID'],
                    'EPISODE_ID': event['EPISODE_ID'],
                    'EVENT_STATE': event['STATE'],
                    'EVENT_COUNTY': event['CZ_NAME'],
                    'EVENT_TYPE': event['EVENT_TYPE'],
                    'EVENT_YEAR': str(event['YEAR']),
                    'EVENT_FIPS': event['FIPS'],
                    'COUNTY_FIPS': county_fips,
                    'COUNTY_NAME': f"{county['NAME']} County",
                    'COUNTY_STATE': county['USPS'],
                    'DISTANCE_KM': distance,
                    'CZ_TYPE': event['CZ_TYPE']
                })
            except (IndexError, KeyError):
                continue
    
    print(f"\nProcessed events summary:")
    for event_type, counts in stats.items():
        print(f"{event_type} events - Processed: {counts['processed']}, "
              f"Skipped: {counts['skipped']}")
    
    return pd.DataFrame(results)

def main():
    # File paths
    gazetteer_path = "/Users/theobaker/Downloads/109final/2024_Gaz_counties_national.txt"
    weather_path = "/Users/theobaker/Downloads/109final/2004-2018_election_day_extreme_events.csv"
    voting_path = "/Users/theobaker/Downloads/109final/nanda_voting_county_2004-2018_01P.csv"

    # Load and clean data
    print("Loading and cleaning data...")
    gazetteer_df, weather_df, voting_df = load_and_clean_data(
        gazetteer_path, weather_path, voting_path
    )
    
    print(f"\nInitial event counts:")
    print(weather_df['CZ_TYPE'].value_counts())
    
    # Process weather events
    print("\nProcessing weather events...")
    results_df = process_weather_events(weather_df, gazetteer_df)
    
    # Merge with voting data
    print("\nMerging with voting data...")
    merged_data = pd.merge(
        results_df,
        voting_df.rename(columns={'stcofips': 'COUNTY_FIPS', 'year': 'VOTING_YEAR'}),
        on='COUNTY_FIPS',
        how='inner'
    )
    
    # Sort and save
    final_data = merged_data.sort_values(['EVENT_ID', 'DISTANCE_KM'])
    final_data.to_csv("/Users/theobaker/Downloads/109final/weather_voting_analysis_improved.csv", index=False)
    
    # Print summary
    print("\nFinal Analysis Summary:")
    print(f"Total rows: {len(final_data)}")
    print(f"Unique events: {len(final_data['EVENT_ID'].unique())}")
    print(f"Unique episodes: {len(final_data['EPISODE_ID'].unique())}")
    print(f"Unique counties: {len(final_data['COUNTY_FIPS'].unique())}")
    print(f"\nEvents by type:")
    print(final_data['CZ_TYPE'].value_counts())
    print(f"\nWeather years: {sorted(final_data['EVENT_YEAR'].unique())}")
    print(f"Voting years: {sorted(final_data['VOTING_YEAR'].unique())}")

if __name__ == "__main__":
    main()