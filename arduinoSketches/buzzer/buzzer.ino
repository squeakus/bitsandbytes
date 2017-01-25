/*
Example 13.0
Drive a piezoelectric buzzer with Arduino
http://tronixstuff.wordpress.com/tutorials > Chapter 13
*/
void setup()
{
     pinMode(10, OUTPUT);   // sets the pin as output
}
void loop()
{
  
     analogWrite(10,255);
     delay(500);
     digitalWrite(10, LOW);
     delay(2000);
}
