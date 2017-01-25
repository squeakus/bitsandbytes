# Shane Lynn 24/03/2014
# @shane_a_lynn
# http://www.shanelynn.ie

import requests
import pandas as pd
import pandas.io.sql as pdsql
from time import sleep, strftime, gmtime
import json
import sqlite3

# define the city you would like to get information from here:
# for full list see http://api.citybik.es
API_URL = "http://api.citybik.es/dublinbikes.json"

#Settings:
SAMPLE_TIME = 120                   # number of seconds between samples
SQLITE = False                      # If true - data is stored in SQLite file, if false - csv.
SQLITE_FILE = "bikedb.db"           # SQLite file to save data in
CSV_FILE = "output1.csv"             # CSV file to save data in

def getAllStationDetails():
    print "\n\nScraping at " + strftime("%Y%m%d%H%M%S", gmtime())

    try:
        # this url has all the details
        decoder = json.JSONDecoder()
        station_json = requests.get(API_URL, proxies='')
        station_data = decoder.decode(station_json.content)
    except:
        print "---- FAIL ----"
        return None

    #remove unnecessary data - space saving
    # we dont need latitude and longitude
    for ii in range(0, len(station_data)):
        del station_data[ii]['lat']
        del station_data[ii]['lng']
        #del station_data[ii]['station_url']
        #del station_data[ii]['coordinates']

    print " --- SUCCESS --- "
    return station_data

def writeToCsv(data, filename="output.csv"):
    """
    Take the list of results and write as csv to filename.
    """
    data_frame = pd.DataFrame(data)
    data_frame['time'] = strftime("%Y%m%d%H%M%S", gmtime())
    data_frame.to_csv(filename, header=True, mode="a")

def writeToDb(data, db_conn):
    """
    Take the list of results and write to sqlite database
    """
    data_frame = pd.DataFrame(data)
    data_frame['scrape_time'] = strftime("%Y%m%d%H%M%S", gmtime())
    pdsql.write_frame(data_frame, "bikedata", db_conn, flavor="sqlite", if_exists="append", )
    db_conn.commit()

if __name__ == "__main__":

    # Create / connect to Sqlite3 database
    conn = sqlite3.connect(SQLITE_FILE) # or use :memory: to put it in RAM
    cursor = conn.cursor()
    # create a table to store the data
    cursor.execute("""CREATE TABLE IF NOT EXISTS bikedata
                      (name text, idx integer, timestamp text, number integer,
                       free integer, bikes integer, id integer, scrape_time text)
                   """)
    conn.commit()

    #run main function
    # we need to run the full collection, parsing, and writing every minute.
    while True:
        station_data = getAllStationDetails()
        if station_data:
            if SQLITE:
                writeToDb(station_data, conn)
            else:
                writeToCsv(station_data, filename=CSV_FILE)

        print "Sleeping for 120 seconds."
        sleep(120)
