import os

import keyboard

# Function to write barcode data to the file


def write_to_file(data):
    with open("out.txt", "a") as f:
        f.write(data + "\n")


# Function to read barcode scanner input


def read_barcode_scanner_input():
    barcode_data = ""
    prev = ""
    while True:
        event = keyboard.read_event()

        if event.event_type == keyboard.KEY_DOWN:
            if event.name == "esc":
                break

            elif event.name == "enter":
                print(f"scan: {barcode_data}")
                write_to_file(barcode_data)
                barcode_data = ""

        else:
            if str(event.name) == "j" and prev == "ctrl":
                pass
            elif len(str(event.name)) == 1:
                barcode_data += str(event.name).upper()
            prev = event.name


if __name__ == "__main__":
    print("Start scanning barcodes (press 'esc' to quit):")

    read_barcode_scanner_input()

    print("Exiting the program...")
