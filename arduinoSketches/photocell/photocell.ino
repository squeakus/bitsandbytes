//int sensorPin = 0;    				// select the input pin for the photocell
int sensorValue = 0;  				// variable to store the value coming from the photocell

void setup() {
  Serial.begin(9600); 				//Set baud rate to 9600 on the Arduino
}

void loop() {
                                          // read the value from the sensor:
  sensorValue = analogRead(A0);  //get the voltage value from input pin
  Serial.println(sensorValue); 		 //print the value to Serial monitor
  delay(2000);                        //delay for 2 seconds
}
