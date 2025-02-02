import os
from datetime import datetime

import pandas as pd
import psycopg2

HOST = "localhost"
DATABASE = "postgres"
USER = "postgres"
PASSWORD = os.getenv("PGPASS")
TABLENAME = "timetest"


def main():
    create_db()

    # Get the current timestamp
    current_timestamp = datetime.now()
    print(current_timestamp)

    # Example usage:
    db_timezone = get_database_timezone()
    print(f"The database timezone is: {db_timezone}")

    add_entry("timestamp plain2", "2023-09-24 14:30:00")  # Replace with the name and timestamp you want to add

    # add_entry(
    #     "timestring2023-09-24 14:30:00+01", "2023-09-24 14:30:00+01"
    # )  # Replace with the name and timestamp you want to add

    df = retrieve_table_contents_raw()
    for row in df:
        print(row)

    print("**********************************************************")
    df = retrieve_table_contents()
    for row in df.iterrows():
        print(row)


def create_db():
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
    )

    # Create a cursor object to execute SQL commands
    cur = conn.cursor()

    # Define the SQL command to create the table
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLENAME} (
        id SERIAL PRIMARY KEY,
        username VARCHAR(255),
        timestamp TIMESTAMP,
        timestamp_tz TIMESTAMPTZ
    );
    """

    # Execute the SQL command
    cur.execute(create_table_sql)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Table created successfully!")


def retrieve_table_contents():
    try:
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=HOST,
            database=DATABASE,
            user=USER,
            password=PASSWORD,
        )

        # Create a cursor object to execute SQL commands
        cur = conn.cursor()

        # Define the SQL query to retrieve data from the table
        select_query = f"SELECT * FROM {TABLENAME};"  # Replace with your table name
        df = pd.read_sql_query(select_query, conn)
        # Execute the SQL query
        conn.close()

        # Return the retrieved data
        return df

    except (Exception, psycopg2.Error) as error:
        print(f"Error retrieving data: {error}")
        return None


def retrieve_table_contents_raw():
    try:
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=HOST,
            database=DATABASE,
            user=USER,
            password=PASSWORD,
        )

        # Create a cursor object to execute SQL commands
        cur = conn.cursor()

        # Define the SQL query to retrieve data from the table
        select_query = f"SELECT * FROM {TABLENAME};"  # Replace with your table name

        cur.execute(select_query)

        # Fetch all rows from the result set
        rows = cur.fetchall()

        # Close the cursor and the database connection
        cur.close()
        conn.close()

        return rows

    except (Exception, psycopg2.Error) as error:
        print(f"Error retrieving data: {error}")
        return None


def add_entry(name, timestamp):
    try:
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=HOST,
            database=DATABASE,
            user=USER,
            password=PASSWORD,
        )

        # Create a cursor object to execute SQL commands
        cur = conn.cursor()

        # Define the SQL command to insert an entry into the table
        insert_entry_sql = f"""
        INSERT INTO {TABLENAME} (username, timestamp, timestamp_tz)
        VALUES (%s, %s, %s::timestamp AT TIME ZONE 'UTC');
        """

        # Execute the SQL command with the provided data
        cur.execute(insert_entry_sql, (name, timestamp, timestamp))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        print("Entry added successfully!")

    except (Exception, psycopg2.Error) as error:
        print(f"Error adding entry: {error}")


import psycopg2


def get_database_timezone():
    try:
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=HOST,
            database=DATABASE,
            user=USER,
            password=PASSWORD,
        )

        # Create a cursor object to execute SQL commands
        cur = conn.cursor()

        # set the timezone
        # cur.execute("""SET TIME ZONE 'UTC';""")

        # Define the SQL query to retrieve the database timezone
        timezone_query = "SHOW timezone;"

        # Execute the SQL query
        cur.execute(timezone_query)

        # Fetch the result (the database timezone)
        database_timezone = cur.fetchone()[0]

        # Close the cursor and connection
        cur.close()
        conn.close()

        return database_timezone

    except (Exception, psycopg2.Error) as error:
        print(f"Error retrieving database timezone: {error}")
        return None


if __name__ == "__main__":
    main()
