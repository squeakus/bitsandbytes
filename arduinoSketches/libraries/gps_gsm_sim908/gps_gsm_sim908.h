/********************* start of gps_gsm_sim908.h ***********************/
 
/*
 *  by 2013-08-02
 *  test on UNO 
 *  Serial0 to GPS
 *
 */
 
//debug
//#define DEBUG
 
#include <Arduino.h>
 
 
#define gps_enable()    digitalWrite (4, LOW)
#define gps_disable()   digitalWrite (4, HIGH)
 
#define gsm_enable()    digitalWrite (3, LOW)
#define gsm_disable()   digitalWrite (3, HIGH)
 
#define GPS_BUF_SIZE 500
#define GGA_NUM 15
#define RMC_NUM 14
 
//
char *gga_table[GGA_NUM] = {
    "Message ID",           //0
    "UTC Time",             //1
    "Latitude",             //2
    "N/S Indicator",        //3
    "Longitude",            //4
    "E/W Indicator",        //5
    "Position Fix Indicator",   //6
    "Satellites Used",      //7
    "HDOP",             //8
    "MSL Altitude",         //9
    "Units(M)",             //10
    "Geoid Separation",         //11
    "Units",            //12
    "Diff.Ref.Station ID",      //13
    "Checksum",         //14
};
 
//
char *gprmc_table[RMC_NUM] = {
    "Message ID",           //0
    "UTC Time",                     //1
    "Status",                       //2
    "Latitude",                     //3
    "N/S Indicator",                //4
    "Langitude",                    //5
    "E/W Indicator",                //6
    "Speed Over Ground",            //7
    "Course Over Ground",           //8
    "Date",                         //9
    "Magnetic Variation",           //10
    "East/West Indicator",          //11
    "Mode",                         //12
    "Checksum",                     //13
};
 
 
//save data from GPS
uint8_t gps_buf[GPS_BUF_SIZE];
 
//save pointer of gga block
uint8_t* gga_p[GGA_NUM];
uint8_t* gprmc_p[RMC_NUM];
 
// check sum using xor
uint8_t checksum_xor (uint8_t *array, uint8_t leng) {
    uint8_t sum = array[0];
    for (uint8_t i=1; i<leng; i++) {
        sum ^= array[i];
    }
    return sum;
}
 
 
//
void start_gsm () {
    digitalWrite (5, HIGH);
    delay (1500);
    digitalWrite (5, LOW);
    delay (1500);
 
    gsm_enable ();
    gps_disable ();
 
    delay (2000);
    #ifdef DEBUG
    Serial.println ("waiting for GPS! ");
    #endif
}

void start_gps (){
    Serial.println ("AT");
    #ifdef DEBUG
    Serial.println ("Send AT");
    #endif
    delay (1000);
    Serial.println ("AT+CGPSPWR=1");
    #ifdef DEBUG
    Serial.println ("Send AT+CGPSPWR=1");
    #endif
    delay (1000);
    Serial.println ("AT+CGPSRST=1");
    #ifdef DEBUG
    Serial.println ("Send AT+CGPSRST=1");
    #endif
    delay (1000);
 
    gsm_disable ();
    gps_enable ();
 
    delay (2000);
    #ifdef DEBUG
    Serial.println ("$GPGGA statement information: ");
    #endif
}
 
// read data to gps_buf[] from GPS
static int gps_read () {
    uint32_t start_time = millis ();
    while (!Serial.available ()) {
        if (millis() - start_time > 1500) {
            #ifdef DEBUG
            Serial.println ("restart GPS......");
            #endif
            start_gps ();
        }
    }
    for (int i=0; i<GPS_BUF_SIZE; i++) {
        delay (7);
        if (Serial.available ()) {
            gps_buf [i] = Serial.read ();
        } else {
            #ifdef DEBUG
            Serial.print ("read ");
            Serial.print (i);
            Serial.println (" character");
            #endif
            return 1;
        }
    }
    #ifdef DEBUG
    Serial.println ("error! data is so big!");
    #endif
    return 0;
}
 
//test head of gps_buf[] if is "$GPGGA" or not
static int is_GPGGA () {
    char gga_id[7] = "$GPGGA";
    for (int i=0; i<6; i++)
        if (gga_id[i] != gps_buf[i])
            return 0;
    return 1;
}
 
//
static uint8_t get_gga_leng () {
    uint8_t l;
    for (l=0; l<GPS_BUF_SIZE && gps_buf[l] != 0x0d ; l++);
    return l;
}
 
// build gga_p[] by gps_buf
static void build_gga_p () {
    int p,b;
    for (p=b=0; p<GGA_NUM && b<GPS_BUF_SIZE; p++,b++) {
        gga_p[p] = (gps_buf+b);//
        if (gps_buf[b] == ',') 
            continue;
        for (b++; b<GPS_BUF_SIZE && gps_buf[b]!=','; b++);
    }
}
 
//test if fix
int gps_gga_is_fix (void) {
    if (gga_p[6][0] == '1')
        return 1;
    else
        return 0;
}
 
//get gga checksum
static uint8_t gps_gga_checksum () {
    uint8_t sum = 0;
    if (gga_p[14][0] != '*')
        return 0;
    if (gga_p[14][2] >= '0' && gga_p[14][2] <= '9')
        sum = gga_p[14][2] - '0';
    else
        sum = gga_p[14][2] - 'A' + 10;
    if (gga_p[14][1] >= '0' && gga_p[14][1] <= '9')
        sum += (gga_p[14][1] - '0') * 16;
    else
        sum += (gga_p[14][1] - 'A' + 10) * 16;
    return sum;
}
 
//check sum of gga
static int checksum_gga () {
    uint8_t sum = checksum_xor (gps_buf+1, get_gga_leng ()-4);
    return sum - gps_gga_checksum ();
}
 
// set gga, change ',' to '\0'
static void gps_gga_set_str () {
    int i;
    for (i=0; gps_buf[i] != 0x0d && i<GPS_BUF_SIZE; i++)
        if (gps_buf[i] == ',')
            gps_buf[i] = '\0';
    //gps_buf[i] = '\0';
}
 
//
int gps_get_gga (void) {
    int stat = 0;
    if (gps_read ()) {
        if (is_GPGGA ()) {
            build_gga_p (); // build *gga_p[] by gps_buf
            gps_gga_set_str ();
            if (checksum_gga () == 0)
                stat = 0;
            else
                stat = 1;
        } else
            stat = 2;
    } else
        stat = 3;
 
    return stat;
}
 
 
//get UTC second
uint8_t gps_gga_utc_ss () {
    return (gga_p[1][4]-'0')*10+gga_p[1][5]-'0';
}
 
//get UTC minute
uint8_t gps_gga_utc_mm () {
    return (gga_p[1][2]-'0')*10+gga_p[1][3]-'0';
}
 
//get UTC hour
uint8_t gps_gga_utc_hh () {
    return (gga_p[1][0]-'0')*10+gga_p[1][1]-'0';
}
 
//return UTC Time string, hhmmss
char* gps_gga_utc_s () {
    return (char*)gga_p[1];
}
 
//get latitude
double gps_gga_lat () {
    return atof ((char*)gga_p[2]);
}
 
//get latitude
char* gps_gga_lat_s () {
    return (char*)gga_p[2];
}
 
//get longitude
double gps_gga_long () {
    return atof ((char*)gga_p[4]);
}
 
//get longitude
char* gps_gga_long_s () {
    return (char*)gga_p[4];
}
 
//get HDOP
double gps_gga_HDOP () {
    return atof ((char*)gga_p[8]);
}
 
//get HDOP
char* gps_gga_HDOP_s () {
    return (char*)gga_p[8];
}
 
//get N/S
char* gps_gga_NS () {
    return (char*)gga_p[3];
    /*
    if (gga_p[3][0] == '\0')
        return '0';
    else if (gga_p[3][0] == 'N' || gga_p[3][0] == 'S')
        return gga_p[3][0];
    else 
        return '?';
        */
}
 
//get E/W
char* gps_gga_EW () {
    return (char*)gga_p[5];
    /*
    if (gga_p[5][0] == '\0')
        return '0';
    else if (gga_p[5][0] == 'E' || gga_p[5][0] == 'W')
        return gga_p[5][0];
    else 
        return '?';
        */
}
 
//
double gps_gga_MSL () {
    return atof ((char*)gga_p[9]);
}
 
//
char* gps_gga_MSL_s () {
    return (char*)gga_p[9];
}
 
//get gpggpa Geoid Separation
double gps_gga_geoid_sep () {
    return atof ((char*)gga_p[11]);
}
 
//get gpggpa Geoid Separation
char* gps_gga_geoid_sep_s () {
    return (char*)gga_p[11];
}
 
#ifdef DEBUG
//
void gps_gga_print () {
    for (int i=0; i<GPS_BUF_SIZE && gps_buf[i]!=0xd; i++)
        Serial.print ((char)gps_buf[i]);
    Serial.println ();
}
#endif
 
/*
void send_string (char* numble, char*string) {
        char num_buf[25];
        sprintf (num_buf, "AT+CMGS=\"%s\"", numble);
        gsm_enable ();
        gps_disable ();
        delay (2000);
        Serial.println ("AT");
        delay (200);
        Serial.println ("AT");
        delay (200);
        Serial.println ("AT+CMGF=1");
        delay (200);
        Serial.println (num_buf);
        delay (200);
        Serial.println (string);
        Serial.write (26);
}
*/
 
// set mobile numble 
void gsm_set_numble (char *numble) {
        char num_buf[25];
        sprintf (num_buf, "AT+CMGS=\"%s\"", numble);
        gsm_enable ();
        gps_disable ();
        delay (2000);
        Serial.println ("AT");
        delay (2000);
        Serial.println ("AT");
        delay (2000);
        Serial.println ("AT+CMGF=1");
        delay (2000);
        Serial.println (num_buf);
        delay (2000);
}
 
// send message to mobile
void gsm_send_message (char *message) {
    Serial.println (message);
}
 
//
void gsm_end_send () {
    Serial.write (26);
    delay (200);
    gsm_disable ();
    gps_enable ();
    delay (2000);
}
 
 
//
void gps_init () {
    pinMode (3, OUTPUT);
    pinMode (4, OUTPUT);
    pinMode (5, OUTPUT);
 
}
 
/********************* end of gps_gsm_sim908.h ***********************/
