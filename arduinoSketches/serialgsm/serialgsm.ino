 // Product name: GPS/GPRS/GSM Module V3.0
// # Product SKU : TEL0051
// # Version     : 0.1
 
// # Description:
// # The sketch for driving the gsm mode via the USB interface
 
// # Steps:
// #        1. Turn the S1 switch to the Prog(right side)
// #        2. Turn the S2 switch to the USB side(left side)
// #        3. Set the UART select switch to middle one.
// #        4. Upload the sketch to the Arduino board(Make sure turn off other Serial monitor )
// #        5. Turn the S1 switch to the comm(left side)       
// #        6. RST the board 
 
// #        wiki link- http://www.dfrobot.com/wiki/index.php/GPS/GPRS/GSM_Module_V3.0_(SKU:TEL0051)
 
void setup()
{
  serial.begin(9600)
  //Init the driver pins for GSM function
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  //Output GSM Timing
  digitalWrite(5, HIGH);
  delay(2000);
  digitalWrite(5, LOW);
  Serial.write("set up and awaiting input");
}
void loop()
{
  // Use these commands instead of the hardware switch 'UART select' in order to enable each mode
  // If you want to use both GMS and GPS. enable the required one in your code and disable the other one for each access.
  digitalWrite(3, LOW); //enable GSM TX、RX
  delay(500);
  digitalWrite(4, HIGH); //disable GPS TX、RX
  delay(500);
}
