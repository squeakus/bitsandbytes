#include <SPI.h>
#include <LoRa.h>
#include<Arduino.h>
#include "SSD1306.h"

#define SCK 5 // GPIO5 - SX1278's SCK serial clock
#define MISO 19 // GPIO19 - SX1278's MISO
#define MOSI 27 // GPIO27 - SX1278's MOSI
#define SS 18 // GPIO18 - SX1278's CS chip select
#define RST 14 // GPIO14 - SX1278's RESET
#define DI0 26 // GPIO26 - SX1278's IRQ (interrupt request)
#define BAND 868E6 // 915E6


String rssi = "RSSI -";
String packSize = "-";
String content = "";
String packet;
String result;

SSD1306 display (0x3c, 4, 15);

void draw_msg(String msg, int line) {
  int spacing = line * 15;
  display.drawString (0, spacing, msg);
}

void loraData() {
  display.clear();
  draw_msg(rssi, 0);
  draw_msg("Receive: " + packSize + " bytes", 1);
  draw_msg(content, 2);
  display.display();
}

void setup () {
  pinMode (16, OUTPUT);
  digitalWrite (16, LOW); // set GPIO16 low to reset OLED
  delay (50);
  digitalWrite (16, HIGH); // while OLED is running, GPIO16 must go high,
  
  Serial.begin (9600);
  while (! Serial);
  Serial.println ();
  Serial.println ("LoRa Receiver Callback");
  SPI.begin (SCK, MISO, MOSI, SS);
  LoRa.setPins (SS, RST, DI0);
  if (! LoRa.begin (868E6)) {
    Serial.println ("Starting LoRa failed!");
    while (1);
  }
  LoRa.receive ();
  Serial.println ("init ok");
  display.init ();
  display.flipScreenVertically ();
  display.setFont (ArialMT_Plain_10);
  
  delay (1500);
}

void loop () {
  int packetSize = LoRa.parsePacket ();
  if (packetSize) {
      packet = "";
      packSize = String (packetSize, DEC);
      for (int i = 0; i < packetSize; i++)  rssi = "RSSI" + String (LoRa.packetRssi(), DEC);
      result = "";
      while (LoRa.available()) {
        result.concat((char)LoRa.read());
      }
      content = "content: " +  result;
      loraData();
  }
  delay (10);
}
