void setup() {                
  // initialize the digital pin as an output.
  // Pin 13 has an LED connected on most Arduino boards:
  pinMode(9, OUTPUT);     
}

void loop() {
  tone(9, 13);   // set the LED on
  delay(1000);              // wait for a second
  tone(9,14);    // set the LED off
  delay(1000);              // wait for a second
}
