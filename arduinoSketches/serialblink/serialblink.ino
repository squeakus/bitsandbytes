/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.
 
  This example code is in the public domain.
 */
 
// Pin 13 has an LED connected on most Arduino boards.
// give it a name:
int led = 13;
int state; 
// the setup routine runs once when you press reset:
void setup() {                
  // initialize the digital pin as an output.
  pinMode(led, OUTPUT);
  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
void loop() {
  if(Serial.available() > 0){
    if (Serial.peek() == 'c'){
      Serial.read();
      state = Serial.parseInt();
      digitalWrite(led, state);
    }
    while (Serial.available() > 0){
      Serial.read();
    }
  }  
}
