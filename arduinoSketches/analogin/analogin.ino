/*
  AnalogReadSerial
 Reads an analog input on pin 0, prints the result to the serial monitor 
 
 This example code is in the public domain.
 */

void setup() {
  /*In the program below, the only thing that you do will in the setup 
  function is to begin serial communications, at 9600 bits of data per
  second, between your Arduino and your computer with the command:*/
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(A0);
  Serial.println(sensorValue);
}
