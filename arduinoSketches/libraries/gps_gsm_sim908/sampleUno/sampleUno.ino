#include "gps_gsm_sim908Serial0.h"
 
void setup () {
  gps_init ();    //init GPS pin
  Serial.begin (9600);    //serial0 connect computer
  start_gps ();   //open GPS
}
 
void loop () {
  int stat = gps_get_gga ();  // read data from GPS, return 0 is ok

  if (stat == 0 || stat == 1) {
    if (gps_gga_is_fix ()) {    //true if fix
      gsm_set_numble ("188*****244"); // change it to your receiver phone number 
      gsm_send_message (gps_gga_utc_s ());
      gsm_send_message (gps_gga_EW ());
      gsm_send_message (gps_gga_NS ());
      gsm_send_message (gps_gga_lat_s ());
      gsm_send_message (gps_gga_long_s ());
      gsm_end_send ();
      while (1);
    }
  }
 
  switch (stat) {
  case 0:
#ifdef DEBUG
    Serial.println ("data checksum is ok");
#endif
    break;
  case 1:
#ifdef DEBUG
    Serial.println ("GPGGA ID is error!");
#endif
    break;
  case 2:
#ifdef DEBUG
    Serial.println ("data is error!");
#endif
    break;
  }
 
#ifdef DEBUG
  Serial.println ("$GPGGA data:");
  gps_gga_print ();   //for test
#endif
  /*
    if (gps_gga_is_fix () == 0) //check if is fix
   Serial.println ("can't fix! please go outside!");
   else {
   Serial.println ("ok! is fix!");
    
   Serial.println ("gps_gga_utc_hh ()");
   Serial.println (gps_gga_utc_hh ());
   Serial.println ("gps_gga_utc_mm ()");
   Serial.println (gps_gga_utc_mm ());
   Serial.println ("gps_gga_utc_ss ()");
   Serial.println (gps_gga_utc_ss ());
    
   Serial.println ("gps_gga_NS ()");
   Serial.println (gps_gga_NS (), 6);
   Serial.println ("gps_gga_EW ()");
   Serial.println (gps_gga_EW (), 6);
    
   Serial.println ("gps_gga_lat ()");
   Serial.println (gps_gga_lat (), 6);
   Serial.println ("gps_gga_long ()");
   Serial.println (gps_gga_long (), 6);
   Serial.println ("gps_gga_HDOP ()");
   Serial.println (gps_gga_HDOP (), 6);
   Serial.println ("gps_gga_MSL ()");
   Serial.println (gps_gga_MSL (), 6);
   Serial.println ();
   }
   */
}