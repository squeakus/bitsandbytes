// Product name: GPS/GPRS/GSM Module V3.0
// # Product SKU : TEL0051
  
// # Description:
// # The sketch for controling the GSM/GPRS/GPS module via SMS.
// # Steps:
// #        1. Turn the S1 switch to the Prog(right side)
// #        2. Turn the S2 switch to the USB side(left side)
// #        3. Set the UART select switch to middle one.
// #        4. Upload the sketch to the Arduino board(Make sure turn off other Serial monitor )
// #        5. Turn the S1 switch to the comm(left side)    
// #        6. Turn the S2 switch to the Arduino(right side)       
// #        7. RST the board until the START led is on(make sure you have >6V power supply)
// #        8. Plug the long side of LED into pin 13 and short side into GND
// #        9. Start sending "LH" and "LL" to your board to turn LED on and off. 
 
/*
 *  created:    2013-11-14
 *  by:     Grey
 *  Version:    0.3
 *  Attention: if you send the wrong SMS command to the module, just need to press RST.
 *  This version can't watch the module status via the serial monitor, it only display the Arduino command.
 *  If you want to watch the status,use the SoftwareSerial or the board with another serial port plese.
 */
  
byte gsmDriverPin[3] = {
  3,4,5};//The default digital driver pins for the GSM and GPS mode
//If you want to change the digital driver pins
//or you have a conflict with D3~D5 on Arduino board,
//you can remove the J10~J12 jumpers to reconnect other driver pins for the module!
int ledpin = 13;
char inchar;
void setup()
{    
  //Init the driver pins for GSM function
  for(int i = 0 ; i < 3; i++){
    pinMode(gsmDriverPin[i],OUTPUT);
  }
  pinMode(ledpin,OUTPUT);
  Serial.begin(9600);                                      //set the baud rate
  digitalWrite(5,HIGH);                                     //Output GSM Timing 
  delay(1500);
  digitalWrite(5,LOW);  
  digitalWrite(3,LOW);                                      //Enable the GSM mode
  digitalWrite(4,HIGH);                                     //Disable the GPS mode
  delay(10000);
  Serial.println("AT+CMGD=1,4");                           //Delete all SMS in box
}

void makecall(String phonenumber){
     Serial.println("AT");//Send AT command     
   delay(2000);
   Serial.println("AT");   
   delay(2000);
  //Make a phone call
   Serial.println("ATD"+phonenumber+";");//Change the receiver phone number
   delay(10000);
   Serial.println("ATH");
   digitalWrite(ledpin,LOW);   
}

void sendtext(){
   Serial.println("AT"); //Send AT command  
  delay(2000);
  Serial.println("AT");   
  delay(2000);
  //Send message
  Serial.println("AT+CMGF=1");
  delay(2000);
  Serial.println("AT+CMGS=\"0863257989\"");//Change the receiver phone number
  delay(2000);
  Serial.print("HELLO");//the message you want to send
  delay(2000);
  Serial.print("\nhow are you?");//the message you want to send
  delay(2000);
  Serial.write(26); 
}

void loop()
{  
  if(Serial.available()>0)
  {
    inchar=Serial.read();

    if(inchar=='T')
    {
      delay(10);
      inchar=Serial.read();
      if (inchar=='I')                                      //When the GSM module get the message, it will display the sign '+CMTI "SM", 1' in the serial port
      {      
        delay(10);
        Serial.println("AT+CMGR=1");                       //When Arduino read the sign, send the "read" AT command to the module
        delay(10);
      }
    }
    else if (inchar=='L')
    {
      delay(10);
      inchar=Serial.read(); 
      if (inchar=='H')                                     //Thw SMS("LH") was display in the Serial port, and Arduino has recognize it.
      {
        delay(10);
        digitalWrite(ledpin,HIGH);                         //Turn on led 
        delay(50);        
        Serial.println("AT+CMGD=1,4");                    //Delete all message
        delay(500);
        makecall("0863257989");
      }
      if (inchar=='L')                                    //Thw SMS("LH") was display in the Serial port, and Arduino has recognize it.
      {
        delay(10);
        digitalWrite(ledpin,LOW);                         //Turn off led 
        delay(50);
        Serial.println("AT+CMGD=1,4");                   //Delete all message
        delay(500);
        sendtext();
      }
    }
  }
}
