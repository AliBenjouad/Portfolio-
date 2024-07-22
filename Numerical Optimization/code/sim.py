import unittest
import math
import csv
import random
from simanneal import Annealer
from deap import base, creator, tools, algorithms
import logging
import numpy as np

# Setting up logging for debugging purposes
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

#Uncomment this and comment the one above for way more extensive loggings for debugging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# GA setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# constants that define the likely hood of two individuals having crossover
# performed and the probability that a child will be mutated. needed for the
# DEAP library
CXPB = 0.5
MUTPB = 0.2
NGEN = 40

# the unit tests to check that the simulation has been implemented correctly
class UnitTests (unittest.TestCase):
    # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
    def testReadCSV(self):
        # read in the locations file
        rows = readCSVFile('track-locations.csv')

        # test that the corners and a middle value are read in correctly
        self.assertEqual('circuit', rows[0][0])
        self.assertEqual('Dec Temp', rows[0][14])
        self.assertEqual('Yas Marina', rows[22][0])
        self.assertEqual('26', rows[22][14])
        self.assertEqual('27', rows[11][8])
    
    # this will test to see if the column conversion works. here we will convert the latitude column and will test 5 values
    # as we are dealing with floating point we will use almost equals rather than a direct equality
    def testColToFloat(self):
        # read in the locations file and convert the latitude column to floats
        rows = readCSVFile('track-locations.csv')
        convertColToFloat(rows, 1)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(26.0325, rows[1][1], delta=0.0001)
        self.assertAlmostEqual(24.4672, rows[22][1], delta=0.0001)
        self.assertAlmostEqual(40.3725, rows[4][1], delta=0.0001)
        self.assertAlmostEqual(30.1327, rows[18][1], delta=0.0001)
        self.assertAlmostEqual(25.49, rows[17][1], delta=0.0001)

    # this will test to see if the column conversion to int works. here we will convert one of the temperature columns and will
    # test 5 values to see that it worked correctly
    def testColToInt(self):
        # read in the locations file and convert the first of the temperature columns to ints
        rows = readCSVFile('track-locations.csv')
        convertColToInt(rows, 3)

        # check that the values are converted correctly
        self.assertEqual(20, rows[1][3])
        self.assertEqual(24, rows[22][3])
        self.assertEqual(4, rows[11][3])
        self.assertEqual(9, rows[16][3])
        self.assertEqual(23, rows[5][3])
    
    # this will test to see if the file conversion overall is successful for the track locations
    # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
    def testReadTrackLocations(self):
        # read in the locations file
        rows = readTrackLocations()

        # check the name, latitude, and final temp of the first race
        self.assertEqual(rows[0][0], 'Bahrain International Circuit')
        self.assertEqual(int(rows[0][14]), 22)
        self.assertAlmostEqual(rows[0][1], 26.0325, delta=0.0001)

        # check the name, longitude, and initial temp of the last race        
        self.assertEqual(rows[21][0], 'Yas Marina')
        self.assertEqual(rows[21][3], 24)
        self.assertAlmostEqual(rows[21][2], 54.603056, delta=0.0001)
    
    # tests to see if the race weekends file is read in correctly
    def testReadRaceWeekends(self):
        # read in the race weekends file
        weekends = readRaceWeekends()

        # check that bahrain is weekend 9 and abu dhabi is weekend 47
        self.assertEqual(weekends[0], 9)
        self.assertEqual(weekends[21], 47)

        # check that hungaroring is weekend 29
        self.assertEqual(weekends[10], 29)
    
    # tests to see if the sundays file is read in correctly
    def testReadSundays(self):
        # read in the sundays file and get the map of sundays back
        sundays = readSundays()

        # check to see the first sunday is january and the last sunday is december
        self.assertEqual(sundays[0], 0)
        self.assertEqual(sundays[51], 11)

        # check a few other random sundays
        self.assertEqual(sundays[10], 2)
        self.assertEqual(sundays[20], 4)
        self.assertEqual(sundays[30], 6)
        self.assertEqual(sundays[40], 9)

    # this will test to see if the haversine function will work correctly we will test 4 sets of locations
    def testHaversine(self):
        # read in the locations file with conversion
        rows = readTrackLocations()

        # check the distance of Bahrain against itself this should be zero
        self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)
        
        # check the distance of Bahrain against Silverstone this should be 5158.08 km
        #Adjust delta for realistic tolerance
        self.assertAlmostEqual(haversine(rows, 0, 9), 5158.08, delta=1.0)
        # check the distance of silverstone against monza this should be 1039.49 Km
        self.assertAlmostEqual(haversine(rows, 13, 9), 1039.49, delta=0.01)

        # check the distance of monza to the red bull ring this should be 455.69 Km
        self.assertAlmostEqual(haversine(rows, 13, 8), 455.69, delta=0.01)
    
    # will test to see if the season distance calculation is correct using the 2023 calendar
    def testDistanceCalculation(self):
        # read in the locations & race weekends, generate the weekends, and calculate the season distance
        tracks = readTrackLocations()
        weekends = readRaceWeekends()
        
        # calculate the season distance using silverstone as the home track as this will be the case for 8 of the teams we will use monza
        # for the other two teams.
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 9), 185874.8866, delta=0.0001)
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 13), 179336.2663, delta=0.0001)
    
    # will test that the temperature constraint is working this should fail as azerbijan should fail the test
    def testTempConstraint(self):
        # load in the tracks, race weekends, and the sundays
        tracks = readTrackLocations()
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 43, 30, 37, 21, 40, 34, 22, 35, 29, 26, 27, 24, 44, 42, 46, 18, 38, 13, 17, 47]
        sundays = readSundays()

        # the test with the default calender should be false because of azerbaijan
        self.assertEqual(checkTemperatureConstraint(tracks, weekends1, sundays), False)
        self.assertEqual(checkTemperatureConstraint(tracks, weekends2, sundays), True)
    
    # will test that we can detect four race weekends in a row.
    def testFourRaceInRow(self):
        # weekend patterns the first does not have four in a row the second does
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 41, 42, 43, 44, 46, 47]

        # the first should pass and the second should fail
        self.assertEqual(checkFourRaceInRow(weekends1), False)
        self.assertEqual(checkFourRaceInRow(weekends2), True)
    
    # will test that we can detect a period for a summer shutdown in july and/or august
    def testSummerShutdown(self):
        # weekend patterns the first has a summer shutdown the second doesn't
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 28, 30, 32, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]

        # the first should pass and the second should fail
        self.assertEqual(checkSummerShutdown(weekends1), True)
        self.assertEqual(checkSummerShutdown(weekends2), False)


# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home
def calculateSeasonDistance(tracks, weekends, home):
    """
    Calculate the total season travel distance for a given sequence of race weekends

    Args:
    - tracks (list): List of track information
    - weekends (list): List of weekend numbers representing race weekends
    - home (int): Index of the home track location in the 'tracks' list

    Returns:
    - float: Total season travel distance in kilometers
    """
    logging.info(f"Starting calculateSeasonDistance with home track index: {home}")

    # Define the mapping from week numbers to track indices
    week_to_track_index = {
        9: 0,   # Bahrain International Circuit
        11: 1,  # Jeddah Corniche Circuit
        13: 2,  # Albert Park
        17: 3,  # Baku City Circuit
        18: 4,  # Miami International Autodromo
        21: 5,  # Monaco
        22: 6,  # Catalunya
        24: 7,  # Circuit Gilles Villeneuve
        26: 8,  # Red Bull Ring
        27: 9,  # Silverstone
        29: 10, # Hungaroring
        30: 11, # Spa Francorchamps
        34: 12, # Zandvoort
        35: 13, # Monza
        37: 14, # Marina Bay Street Circuit
        38: 15, # Suzuka
        40: 16, # Lusail
        42: 17, # COTA
        43: 18, # Autodromo Hermanos Rodriguez
        44: 19, # Interlagos
        46: 20, # Las Vegas Strip Circuit
        47: 21  # Yas Marina
    }

    total_distance = 0
    current_location = home

    for i, week_number in enumerate(weekends):
        logging.info(f"Processing weekend {i + 1}/{len(weekends)}: week number {week_number}")

        # Retrieve the correct track index using the mapping
        if week_number in week_to_track_index:
            week_index = week_to_track_index[week_number]
        else:
            logging.error(f"Week number {week_number} does not have a corresponding track index.")
            continue

        logging.info(f"Current location index: {current_location}, Next race location index: {week_index}")
        distance = haversine(tracks, current_location, week_index)
        logging.info(f"Distance from {tracks[current_location][0]} to {tracks[week_index][0]}: {distance} km")

        total_distance += distance
        current_location = week_index

        logging.debug(f"Total distance after weekend {i + 1}: {total_distance} km")

        # Check if next race is in a non-consecutive weekend
        if i < len(weekends) - 1 and (weekends[i + 1] - week_number) > 1:
            logging.info("Next race is not in a consecutive weekend, returning to home track")
            return_distance = haversine(tracks, week_index, home)
            logging.info(f"Return distance from race index {week_index} to home index {home}: {return_distance} km")

            total_distance += return_distance
            current_location = home

            logging.debug(f"Total distance after returning to home: {total_distance} km")

    # Return to home after the last race
    logging.info(f"Calculating return distance to home after the last race from index {current_location}")
    final_return_distance = haversine(tracks, current_location, home)
    logging.info(f"Final return distance: {final_return_distance} km")

    total_distance += final_return_distance
    logging.info(f"Total season distance calculated: {total_distance} km")

    return total_distance

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will check to see if there is anywhere in our weekends where four races appear in a row. True indicates that we have four in a row
def checkFourRaceInRow(weekends):
    logging.info("Checking for four consecutive race weekends.")

    for i in range(len(weekends) - 3):
        logging.info(f"Checking weekends {i+1} to {i+4} for consecutive races.")
        if weekends[i+3] - weekends[i] == 3:  # Checking four consecutive weekends
            logging.info(f"Found four consecutive race weekends: {weekends[i]} to {weekends[i+3]}.")
            return True

    logging.info("No four consecutive race weekends found.")
    return False

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of 20 degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, sundays):
    """
    Check temperature constraints for the race weekends

    Args:
    - tracks (list): List of track information
    - weekends (list): List of weekend numbers representing race weekends
    - sundays (list): List of months corresponding to Sundays for each race weekend

    Returns:
    - bool: True if all temperature constraints are satisfied, False otherwise
    """
    logging.info("Starting to check temperature constraints for the race weekends.")
    # Mapping from the weekend number to the corresponding track index
    weekend_to_track_index = {weekend: index for index, weekend in enumerate(weekends)}

    logging.debug(f"Tracks list: {tracks}")
    logging.debug(f"Weekends list: {weekends}")
    logging.debug(f"Sundays list: {sundays}")
    logging.debug(f"Weekend to Track Index Mapping: {weekend_to_track_index}")

    try:
        for week_number in weekends:
            # Retrieve the corresponding track index using the mapping
            track_index = weekend_to_track_index[week_number]
            # Retrieve the month index for the given week number
            month_index = sundays[week_number - 1]  # Adjusting for 0-based indexing

            # Access the temperature for the given track and month
            temperature = float(tracks[track_index][month_index + 3])  # Offset by 3 for non-temperature columns

            # Log the temperature information for debugging
            logging.info(f"Weekend {week_number} (Track: {tracks[track_index][0]}, Month Index: {month_index}) has temperature {temperature} degrees.")

            if temperature < 20 or temperature > 35:
                logging.warning(f"Weekend {week_number}: temperature out of bounds at {temperature} degrees.")
                return False

        logging.info("All temperature constraints are satisfied.")
        return True
    except Exception as e:
        logging.error(f"Error in checking temperature constraints: {e}")
        return False



# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. 
def checkSummerShutdown(weekends):
    """
    Check for a summer shutdown period in July or August.
    Args:
    - weekends : List of weekend numbers representing race weekends
    Returns:
    - bool: True if a summer shutdown period is found, False otherwise
    """
    logging.info("Checking for a summer shutdown period in July or August.")
    # July and August are represented by 6 and 7
    for i in range(len(weekends) - 1):
        logging.info(f"Checking if weekend {weekends[i]} is in July or August.")
        if weekends[i] > 26 and weekends[i] < 35:  # Checking if weekend is in July or August
            logging.info(f"Weekend {weekends[i]} is in July or August.")
            if weekends[i+1] - weekends[i] >= 3:  # Checking for a three-week gap
                logging.info(f"Found a three-week gap after weekend {weekends[i]}.")
                return True

    logging.info("No summer shutdown period found.")
    return False




# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will take in the set of rows and will convert the given column index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first row
def convertColToFloat(rows, column_index):
    logging.info(f"Attempting to convert column {column_index} to float")
    try:
        for row in rows[1:]:  # Skipping the header row
            original_value = row[column_index]
            row[column_index] = float(row[column_index])
            logging.info(f"Converted {original_value} to {row[column_index]} (float)")
    except Exception as e:
        logging.error(f"Error in converting column {column_index} to float: {e}")


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_  
# funciton that will take in a set of rows and will convert the given column index into integer values
# this assumes the header in the CSV file is still present so it will skip the first row
def convertColToInt(rows, column_index):
    try:
        for row in rows[1:]:  # Skipping the header row
            row[column_index] = int(row[column_index])
        logging.info(f"Successfully converted column {column_index} to int")
        return True  # Indicate successful conversion
    except Exception as e:
        logging.error(f"Error in converting column {column_index} to int: {e}")
        return False  # Indicate conversion failure


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(tracks, location1, location2):
    """
    Calculate the Haversine distance between two track locations

    Args:
    - tracks (list): List of track information including location and temperature data
    - location1 (int): Index representing the starting point of travel
    - location2 (int): Index representing the ending point of travel

    Returns:
    - float: Haversine distance in kilometers between the specified track locations
    
    Raises:
    - IndexError: If indices are negative or out of valid range
    - ValueError: If latitude or longitude conversion fails
    - Exception: For unexpected errors during calculation
    """
    R = 6371.0000  # The exact arth radius in kilometers
    logging.info(f"Starting Haversine calculation between locations {location1} and {location2}")
    
    # Log the number of tracks to ensure no header row and to check list size
    logging.info(f"Total number of track locations: {len(tracks)}")
    logging.info(f"First track location to verify no header is included: {tracks[0]}")
    
    # Ensure that the indices are 0-based and within the valid range
    if location1 < 0 or location2 < 0:
        logging.error("Negative index error: Indices should be 0-based and non-negative.")
        raise IndexError("Negative index error: Indices should be 0-based and non-negative.")
    
    # Check if the location indices are valid
    if location1 >= len(tracks) or location2 >= len(tracks):
        logging.error(f"Invalid location index: {location1} or {location2}")
        raise IndexError(f"Invalid location index: {location1} or {location2}")

    try:
        # Convert latitude and longitude to radians
        lat1, lon1 = map(math.radians, [float(tracks[location1][1]), float(tracks[location1][2])])
        lat2, lon2 = map(math.radians, [float(tracks[location2][1]), float(tracks[location2][2])])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c

        logging.info(f"Calculated Haversine distance between {tracks[location1][0]} and {tracks[location2][0]}: {distance} km")
        return distance

    except IndexError as e:
        logging.error(f"IndexError in Haversine calculation: {e}")
        raise
    except ValueError as e:
        logging.error(f"ValueError in Haversine calculation: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in Haversine calculation: {e}")
        raise





# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home, sundays):
    """
    Print the race itinerary based on track information, weekends, home location, and Sunday data
    Args:
    - tracks (list): List of track information including location and temperature data
    - weekends (list): List of weekends corresponding to race events
    - home (int): Index representing the home track location
    - sundays (list): List denoting which month each Sunday falls in
    """
    logging.info("Printing the race itinerary.")
    # Adjust home to skip the header row
    current_location = home + 1
    for week_number in weekends:
        # Adjust the week index to skip the header row
        week_index = week_number + 1
        month_index = sundays[week_number] + 3  # Offset by 3 to account for the first three columns

        if current_location != home + 1:
            logging.info(f"Travelling from {tracks[current_location][0]} to home base at {tracks[home + 1][0]}")
            print(f"Travelling from {tracks[current_location][0]} to {tracks[home + 1][0]}")
            current_location = home + 1

        logging.info(f"Weekend {week_number}: Race at {tracks[week_index][0]} with expected temperature {tracks[week_index][month_index]} degrees")
        print(f"Weekend {week_number}: Race at {tracks[week_index][0]}. Expected temperature: {tracks[week_index][month_index]} degrees")
        current_location = week_index


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will read in the race weekends file and will perform all necessary conversions on it
def readCSVFile(file):
    """
    Read and process a CSV file.
    Args:
    - file (str): The path to the CSV file.
    Returns:
    - list: A list of rows, where each row is a list of values from the CSV file.
    """
    logging.info(f"Attempting to read file: {file}")
    # The rows to return
    rows = []
    try:
        # Open the file for reading and give it to the CSV reader
        csv_file = open(file)
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Read in each row and append it to the list of rows
        for row in csv_reader:
            rows.append(row)
            logging.info(f"Read row: {row}")
        # Close the file when reading is finished
        csv_file.close()
        logging.info(f"Successfully read {len(rows)} rows from {file}")
    except Exception as e:
        logging.error(f"Error reading file {file}: {e}")

    # Return the rows at the end of the function
    return rows



# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will read in the sundays file that will map the sundays to a list. each sunday maps to a month. we will need this for temperature comparisons later on
def readSundays():
    """
    Read and process data from the 'sundays.csv' file
    Returns:
    - list: List of Sunday data representing months as integers
    """
    try:
        logging.info("Starting to read and process sundays.csv")
        rows = readCSVFile('sundays.csv')  # Read the CSV file
        if rows:
            # Skip the header when converting and returning values
            convertColToInt(rows, 1)  # Convert the 'month' column to integers
            logging.info("Successfully converted 'month' column to integers in sundays.csv")
            # Extracting only the 'month' column values
            sundays = [row[1] for row in rows[1:]]
            logging.info(f"Extracted sundays data: {sundays}")
            return sundays
        else:
            logging.warning("No data found in sundays.csv")
            return None
    except Exception as e:
        logging.error(f"Error reading sundays.csv: {e}")
        return None


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# function that will read the track locations file and will perform all necessary conversions on it
def readTrackLocations():
    """
    Read and process data from the 'track-locations.csv' file
    Returns:
    - list: List of track locations including latitude, longitude, and temperature data
    """
    try:
        logging.info("Starting to read and process track-locations.csv")
        rows = readCSVFile('track-locations.csv')  # Read the CSV file

        if rows:
            # Logging the headers for clarity
            logging.info(f"Headers in track-locations.csv: {rows[0]}")

            # Perform data type conversion here
            for row in rows[1:]:
                row[1] = float(row[1])  # Convert Latitude to float
                row[2] = float(row[2])  # Convert Longitude to float
                for i in range(3, 15):
                    row[i] = int(row[i])  # Convert temperature values to int
            # Skip the header when returning
            track_locations = rows[1:]
            logging.info("Successfully read track locations")
            # Log the tracks list
            logging.debug(f"'tracks' list: {track_locations}")
            return track_locations
        else:
            logging.warning("No data found in track-locations.csv")
            return None
    except Exception as e:
        logging.error(f"Error reading track-locations.csv: {e}")
        return None


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
def readRaceWeekends():
    """
    Read and process data from the 'race-weekends.csv'
    Returns:
    - list: List of race weekends (week numbers) for the F1 calendar
    """
    logging.info("Starting to read and process race-weekends.csv")
    try:
        file_path = 'race-weekends.csv'  # File path to the CSV file
        rows = readCSVFile(file_path)  # Read the CSV file

        if not rows:
            logging.error("Failed to read any data from race-weekends.csv")
            return None

        logging.info("Successfully read race-weekends.csv")

        # Convert 'week' column to integers
        if not convertColToInt(rows, 1):
            logging.error("Data conversion failed. Check the 'week' column for non-integer values.")
            return None

        # Initialize the race_weekends list with the correct size based on the number of races
        num_races = len(rows) - 1  # Subtract 1 for the header row
        race_weekends = [None] * num_races

        for row in rows[1:]:
            try:
                race_weekend_number = int(row[0])
                week = int(row[1])  # Already converted

                # Populate the race_weekends list for each race weekend
                race_weekends[race_weekend_number - 1] = week

            except ValueError as e:
                logging.error(f"Data conversion error for row {row}: {e}")
                return None
        logging.info("Successfully processed race weekends data.")
        # Log the weekends list
        logging.debug(f"'weekends' list: {race_weekends}")
        return race_weekends
    except Exception as e:
        logging.error(f"Unexpected error reading race-weekends.csv: {e}")
        return None




#=======================================================================================================================================================================
    # Implement the algorithms when the unit tests are succesfull
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
class F1CalendarOptimization(Annealer):
    def __init__(self, state, tracks, fixed_races, weekend_to_track_index):
        """
        Constructor for the F1CalendarOptimization class

        Args:
        - state (list): Current state representing the race order
        - tracks (list): Information about F1 race tracks
        - fixed_races (list): Indices of races that are fixed and cannot be moved.
        - weekend_to_track_index (dict): Mapping of weekend numbers to track indices
        """
        self.tracks = tracks
        self.fixed_races = fixed_races
        self.weekend_to_track_index = weekend_to_track_index
        super(F1CalendarOptimization, self).__init__(state)

    def move(self):
        """
        Generate a move by swapping two random race weekends (excluding fixed races)
        """
        # Select two random indices to swap, excluding fixed races
        a, b = random.sample([i for i in range(len(self.state)) if i not in self.fixed_races], 2)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """
        Calculate the energy (total travel distance) of the current state

        Returns:
        - float: Total travel distance (energy) of the current state
        """
        energy = 0
        for i in range(len(self.state) - 1):
            track_index_1 = self.weekend_to_track_index[self.state[i]]
            track_index_2 = self.weekend_to_track_index[self.state[i + 1]]
            energy += haversine(self.tracks, track_index_1, track_index_2)

        final_track_index = self.weekend_to_track_index[self.state[-1]]
        first_track_index = self.weekend_to_track_index[self.state[0]]
        energy += haversine(self.tracks, final_track_index, first_track_index)

        return energy

    

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# GA Algorithm Class
class F1CalendarGA:
    def __init__(self, tracks, weekend_to_track_index, fixed_races_indices, weekends):
        """
        Constructor for the F1CalendarGA class
        
        Args:
        - tracks (list): Information about F1 race tracks
        - weekend_to_track_index (dict): Mapping of weekend numbers to track indices
        - fixed_races_indices (list): Indices of races that are fixed and cannot be moved
        - weekends (list): List of weekends representing the F1 calendar
        """
        self.tracks = tracks
        self.weekend_to_track_index = weekend_to_track_index
        self.fixed_races_indices = fixed_races_indices
        self.weekends = weekends

    def evalF1Calendar(self, individual):
        """
        Evaluate the fitness of an individual race calendar
        Args:
        - individual (list): List of indices representing the race order
        Returns:
        - tuple: Total travel distance as a single-element tuple
        """
        # Convert indices to weekend numbers
        weekend_numbers = [self.weekends[idx] for idx in individual]
        # Convert weekend numbers to track indices
        track_indices = [self.weekend_to_track_index[weekend] for weekend in weekend_numbers]
        # Calculate total distance
        total_distance = sum(haversine(self.tracks, track_indices[i], track_indices[i + 1]) 
                             for i in range(len(track_indices) - 1))
        return total_distance,

    def setupToolbox(self):
        """
        Set up the toolbox for the genetic algorithm optimization
        Returns:
        - toolbox: A configured toolbox for the genetic algorithm
        """
        toolbox = base.Toolbox()

        # Attribute generator: generate indices based on the length of weekends
        toolbox.register("indices", random.sample, range(len(self.weekends)), len(self.weekends))
            
        # Structure initializers
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evalF1Calendar)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        return toolbox

#------------------------------------------------------------------------------------
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
def SAcases(tracks, weekends, fixed_races_indices, weekend_to_track_index):
    # Create an initial state by copying the original weekends list
    initial_state = weekends.copy()
    # Create an instance of F1CalendarOptimization for Simulated Annealing
    f1_optimization = F1CalendarOptimization(initial_state, tracks, fixed_races_indices, weekend_to_track_index)
    # Set the number of annealing steps (adjust as needed)
    f1_optimization.steps = 100000
    # Perform Simulated Annealing to find the optimized state and total distance
    optimized_state, total_distance = f1_optimization.anneal()
    # Print the optimized race order obtained through Simulated Annealing
    print("\nOptimized Race Order:")
    for week_number in optimized_state:
        track_index = weekend_to_track_index.get(week_number)
        if track_index is not None and track_index < len(tracks):
            print(tracks[track_index][0])
        else:
            print(f"Invalid week number: {week_number}, no corresponding track found.")
    
    # Print the total travel distance achieved by the optimized state
    print("Total Travel Distance: {:.2f} Km".format(total_distance))
    
    # Return the total distance achieved by Simulated Annealing
    return total_distance


#------------------------------------------------------------------------------------
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

def GAcases(tracks, weekends, fixed_races_indices, weekend_to_track_index):
    # Create an instance of the F1CalendarGA class for genetic algorithm optimization
    ga_optimization = F1CalendarGA(tracks, weekend_to_track_index, fixed_races_indices, weekends)
    # Setup the toolbox for genetic algorithm optimization
    toolbox = ga_optimization.setupToolbox()
    # Initialize a population of 300 individuals
    population = toolbox.population(n=300)
    # Create a Hall of Fame to store the best individual
    hof = tools.HallOfFame(1, similar=np.array_equal)

    # Create statistics to track the evolution of the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm using eaSimple
    algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof)

    # Get the best individual and its fitness (distance)
    best_individual = hof[0]
    best_distance = ga_optimization.evalF1Calendar(best_individual)[0]

    # Print the optimized race order obtained through Genetic Algorithm
    print("\nOptimized Race Order (GA):")
    for idx in best_individual:
        week_number = weekends[idx]  # Convert index to weekend number
        track_index = weekend_to_track_index.get(week_number)
        if track_index is not None and track_index < len(tracks):
            print(tracks[track_index][0])
        else:
            print(f"Invalid week number: {week_number}, no corresponding track found.")
    
    # Print the total travel distance achieved by the best individual
    print("Total Travel Distance (GA): {:.2f} Km".format(best_distance))

    # Return the best distance achieved by the genetic algorithm
    return best_distance


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#Main initialization of the script, start with unnit tests, SA, then GA, integrate a sort of comparison mechanism

if __name__ == '__main__':
    #False exit for the script to continue it's execution
    unittest.main(exit=False)
    tracks = readTrackLocations()
    weekends = readRaceWeekends()
    #Fixed races
    fixed_races_indices = [0, 21, 5]  # Bahrain, Abu Dhabi, Monaco

    # Define the weekend to track index mapping
    weekend_to_track_index = {9: 0, 11: 1, 13: 2, 17: 3, 18: 4, 21: 5, 22: 6, 24: 7, 26: 8, 27: 9, 29: 10, 30: 11, 34: 12, 35: 13, 37: 14, 38: 15, 40: 16, 42: 17, 43: 18, 44: 19, 46: 20, 47: 21}

    # Run Simulated Annealing and Genetic Algorithm Cases
    sa_distance = SAcases(tracks, weekends, fixed_races_indices, weekend_to_track_index)
    ga_distance = GAcases(tracks, weekends, fixed_races_indices, weekend_to_track_index)

    # Comparing results
    print("\nComparison of Optimizations:")
    if sa_distance is not None and ga_distance is not None:
        print(f"Simulated Annealing Total Distance: {sa_distance:.2f} Km")
        print(f"Genetic Algorithm Total Distance: {ga_distance:.2f} Km")
    
        if sa_distance < ga_distance:
            print("Simulated Annealing produced a shorter total travel distance.")
        elif ga_distance < sa_distance:
            print("Genetic Algorithm produced a shorter total travel distance.")
        else:
            print("Both algorithms resulted in the same total travel distance.")
    else:
        print("Error in computing distances. Please check the implementations.")