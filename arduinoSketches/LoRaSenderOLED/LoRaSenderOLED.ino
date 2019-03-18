#include "SSD1306.h"
#include <SPI.h>
#include <LoRa.h>

#define SCK 5 // GPIO5 - SX1278's SCK
#define MISO 19 // GPIO19 - SX1278's MISO
#define MOSI 27 // GPIO27 - SX1278's MOSI
#define SS 18 // GPIO18 - SX1278's CS chip select 
#define RST 14 // GPIO14 - SX1278's RESET
#define DI0 26 // GPIO26 - SX1278's IRQ (interrupt request)
#define BAND 868E6 // 915E6

unsigned int counter = 0;

String rssi = "RSSI -";
String packSize = "-";
String packet;


SSD1306 display (0x3c, 4, 15);

void draw_msg(String msg, int line) {
  int spacing = line * 15;
  display.drawString (0, spacing, msg);
}


void setup () {
  pinMode (16, OUTPUT);
  pinMode (2, OUTPUT);
  
  digitalWrite (16, LOW); // set GPIO16 low to reset OLED
  delay (50);
  digitalWrite (16, HIGH); // while OLED is running, GPIO16 must go high
  
  Serial.begin (9600);
  while (! Serial);
  Serial.println ();
  Serial.println ("LoRa Sender Test");
  
  SPI.begin (SCK, MISO, MOSI, SS);
  LoRa.setPins (SS, RST, DI0);
  if (! LoRa.begin (BAND)) {
    Serial.println ("Starting LoRa failed!");
    while (1);
  }
  //LoRa.onReceive(cbk);
// LoRa.receive ();
  Serial.println ("init ok");
  display.init ();
  display.flipScreenVertically ();
  display.setFont (ArialMT_Plain_10);
  delay (1500);
}



void loop () {
  display.clear();
  String message = "Sending packet: " + String(counter);
  draw_msg(message, 0);
  display.display();

  // send packet
  LoRa.beginPacket ();
  LoRa.print ("hello");
  LoRa.print (counter);
  LoRa.endPacket ();

  counter ++;
  digitalWrite (2, HIGH); // turn the LED on (HIGH is the voltage level)
  delay (1000); // wait for a second
  digitalWrite (2, LOW); // turn the LED off by making the voltage LOW
  delay (1000); // wait for a second
}
