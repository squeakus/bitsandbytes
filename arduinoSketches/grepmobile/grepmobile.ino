char inchar;
char phnumber[11]={'0','0','0','0','0','0','0','0','0','\0'};


void setup()
{    
  //Init the driver pins for GSM function
  pinMode(3,OUTPUT);
  pinMode(4,OUTPUT);
  pinMode(5,OUTPUT);

  Serial.begin(9600);      //set the baud rate
  digitalWrite(5,HIGH);    //Output GSM Timing
  delay(1500);
  digitalWrite(5,LOW);  
  digitalWrite(3,LOW);
  digitalWrite(4,HIGH);    // disable the GPS mode
  delay(10000);
  Serial.println("AT+CMGD=1,4");   //Delete all SMS in box
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
   delay(2000);
   Serial.println("AT+CMGD=1,4");   //Delete all SMS in box
}

void sendtext(String number){
   Serial.println("AT"); //Send AT command  
  delay(2000);
  Serial.println("AT");   
  delay(2000);
  //Send message
  Serial.println("AT+CMGF=1");
  delay(2000);
  Serial.println("AT+CMGS=\""+number+"\"");//Change the receiver phone number
  delay(2000);
  Serial.print("Hello!");//the message you want to send
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
    else if (inchar=='3'){
      delay(10);
      inchar=Serial.read(); 
      if (inchar=='5'){
        delay(10);
        inchar=Serial.read(); 
        if (inchar=='3'){
           
           for(int i = 1; i < 10; i++){
             inchar=Serial.read();
             phnumber[i]= inchar;
           }
           Serial.println ("found number!");
           Serial.println (phnumber);
           
        }
      }
    }
         
    else if (inchar=='L')
    {
      delay(10);
      inchar=Serial.read(); 
      if (inchar=='H')                                     //Thw SMS("LH") was display in the Serial port, and Arduino has recognize it.
      {
        delay(500);
        sendtext(phnumber);
        //makecall(phnumber);
      }
    }
  }
}
