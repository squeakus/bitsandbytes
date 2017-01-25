/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.
 
  This example code is in the public domain.
 */

void setup() {                
  // initialize the digital pin as an output.
  // Pin 13 has an LED connected on most Arduino boards:
  pinMode(12, OUTPUT);     
}

void loop() {
  int sensorValue = analogRead(A0);
  Serial.println(sensorValue);
  digitalWrite(12, HIGH);   // set the LED on
  delay(sensorValue);              // wait for a second
  digitalWrite(12, LOW);    // set the LED off
  delay(sensorValue);              // wait for a second
}
