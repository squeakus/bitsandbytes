#include "gps_gsm_sim908.h"
 
boolean sendinfo = false;
boolean gpsinit = true;
char inchar;
char phnumber[11]={'0','0','0','0','0','0','0','0','0','\0'};

void setup () {
  //Init the driver pins for GSM function
  gps_init();  
  Serial.begin (9600);    //serial0 connect computer                     
  //Serial.println("AT+CMGD=1,4");
  delay(5000);
  start_gsm ();
  Serial.println ("AT");
  delay(1000); 
  Serial.println("AT+CMGD=1,4");  
  }
 
 void makecall(String phonenumber){
   gsm_enable ();
   gps_disable ();
   delay (2000);
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
 
void sendcoords(){
  gsm_set_numble (phnumber); // change it to your receiver phone number 
  gsm_send_message ("Hello! The time is:");
  gsm_send_message (gps_gga_utc_s ());
  gsm_send_message ("And my coords are:");
  gsm_send_message (gps_gga_lat_s ());
  gsm_send_message (gps_gga_NS ());
  gsm_send_message (gps_gga_long_s ());
  gsm_send_message (gps_gga_EW ());
  delay (2000);
  Serial.write (26);
  delay (2000);
  //gsm_disable ();
  //gps_enable ();

  //gsm_end_send ();
} 
 
void loop () {
{ 
  if(Serial.available()>0)
  {
    inchar=Serial.read();
   
    // check for messages
    if(inchar=='T')
    { 
      delay(10);
      inchar=Serial.read(); 
      if (inchar=='I')                                      //When the GSM module get the message, it will display the sign '+CMTI "SM", 1' in the serial port
      { 
        //Serial.println("reading message");  
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
    
    // read text message
    else if (inchar=='A')
    { 
      delay(10);
      inchar=Serial.read(); 
      if (inchar=='1')                                     //Thw SMS("LH") was display in the Serial port, and Arduino has recognize it.
      {
        delay(500);
        makecall(phnumber);
        sendinfo = true;
      }
    }
  }
  
  // if text received then send gps coords
  int cnt = 0;
  while(sendinfo){
    cnt++;
    Serial.println("Checking"+ String(cnt));
    if(gpsinit){
      start_gps();
      gpsinit = false;
    }
    
    int stat = gps_get_gga ();  // read data from GPS, return 0 is ok

    if (stat == 0 || stat == 1) {
      if (gps_gga_is_fix ()) {    //true if fix
        sendcoords();
        sendinfo = false;
        gpsinit = true;

      }
    }
  }
}

}
