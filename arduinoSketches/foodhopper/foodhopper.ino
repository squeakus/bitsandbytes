int buttonPin = 2;
int buttonState = 0;
int forwardPin = 9;
int backPin = 10;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(buttonPin, INPUT);
  pinMode(forwardPin, OUTPUT);
  pinMode(backPin, OUTPUT);
}

void loop() {
 
  digitalWrite(backPin, LOW);


  buttonState = digitalRead(buttonPin);
  // Show the state of pushbutton on serial monitor
  Serial.println(buttonState);
  delay(50);

  if (buttonState == HIGH) {
    Serial.println("Forward");
    for (int fadeValue = 100; fadeValue <= 255; fadeValue += 10) {
      // sets the value (range from 0 to 255):
      analogWrite(forwardPin, fadeValue);
      // wait for 30 milliseconds to see the dimming effect
      delay(500);
    } 
    analogWrite(forwardPin, 0);
    Serial.println("Finished");
  }
}
