/*
  LoRa Client Battery Test

  Note: while sending, LoRa radio is not listening for incoming messages.
  Note2: when using the callback method, you can't use any of the Stream
  functions that rely on the timeout, such as readString, parseInt(), etc.

  created 28 April 2017
  adapted from the code provided by Tom Igoe
*/
#include <SPI.h>              // include libraries
#include <LoRa.h>
#include<Arduino.h>
#include "SSD1306.h"

#define SCK 5 // GPIO5 - SX1278's SCK serial clock
#define MISO 19 // GPIO19 - SX1278's MISO
#define MOSI 27 // GPIO27 - SX1278's MOSI
#define CS 18 // GPIO18 - SX1278's CS Radio chip select
#define RST 14 // GPIO14 - SX1278's LoRa radio reset
#define DI0 26 // GPIO26 - SX1278's IRQ (interrupt request) change for your board; must be a hardware interrupt pin
#define BAND 868E6 // Frequencxy band to operate on.
#define uS_TO_S_FACTOR 1000000  /* Conversion factor for micro seconds to seconds */

int lastSleepTime = 0;
int sleepInterval = 30000;
int time_to_sleep = 900;      /* Time ESP32 will go to sleep (in seconds) */

byte msgCount = 0;            // count of outgoing messages
byte localAddress = 0xBB;     // address of this device
byte destination = 0xAA;      // destination to send to
long lastSendTime = 0;        // last send time
int interval = 2000;          // interval between sends
int recipient = 0;          // recipient address
byte sender;            // sender address
float vBat = 0;
byte incomingMsgId;     // incoming msg ID
byte incomingLength;    // incoming msg length
String incoming = "";                 // payload of packet
String message = "None";
bool beddybyes = false;
SSD1306 display (0x3c, 4, 15); // OLED Screen

void init_display() {
  pinMode (16, OUTPUT);
  pinMode (2, OUTPUT);
  digitalWrite (16, LOW); // set GPIO16 low to reset OLED
  delay (50);
  digitalWrite (16, HIGH); // while OLED is running, GPIO16 must go high
  display.init ();
  display.flipScreenVertically ();
  display.setFont (ArialMT_Plain_10);
  delay(1500);
  display.clear();
  draw_msg("LoRa Battery Client", 0);
  display.display();
  delay(1500);
}

void draw_msg(String msg, int line) {
  int spacing = line * 10;
  display.drawString (0, spacing, msg);
}

float average (float * array, int len)  // assuming array is int.
{
  long sum = 0L ;  // sum will be larger than an item, long for safety.
  for (int i = 0 ; i < len ; i++)
    sum += array [i] ;
  return  ((float) sum) / len ;  // average will be fractional, so float may be appropriate.
}

byte battery_level() {
//  float sampleArray[10];
//  int samples = 0;
//
//  while (samples < 10) {
//    delay(100);
//    float val = float(analogRead(34));
//    if (val > 0 ){
//      sampleArray[samples] = val;
//      samples += 1;
//    }
//  }
//  float vAverage = average(sampleArray, 10);
  float vAverage = 1000.0f;
  byte vByte = int(255 * ( vAverage / 4096.0f));
  return vByte;
}
void setup() {
  Serial.begin(9600);                   // initialize serial
  while (!Serial);
  pinMode (34, INPUT); // pin for measuring battery
  init_display();

  // override the default CS, reset, and IRQ pins (optional)
  LoRa.setPins(CS, RST, DI0);// set CS, reset, IRQ pin
  lastSleepTime = millis();
  esp_sleep_enable_timer_wakeup(time_to_sleep * uS_TO_S_FACTOR);
  
  if (!LoRa.begin(BAND)) {             // initialize ratio at 868 MHz
    Serial.println("LoRa init failed. Check your connections.");
    while (true);                       // if failed, do nothing
  }

  LoRa.onReceive(onReceive);
  LoRa.receive();
  Serial.println("LoRa init succeeded.");
  byte vByte = battery_level();
  vBat= (4.2 * float(vByte)) / 255.0f; 
}

void loop() {
  display.clear();
  
  draw_msg("Client " + String(localAddress, HEX) + " is listening", 0);
  draw_msg("battery: " + String(vBat) + "v", 1);

  if (incoming != "") {
    String outgoing = "Ack"  + incoming;

    draw_msg("Received: " + String(sender, HEX) + " MsgID: " + String(incomingMsgId), 2);
    draw_msg("Msg:" + incoming, 3);
    draw_msg("Ret:" + outgoing, 4);
    draw_msg("RSSI: " + String(LoRa.packetRssi()) + " S/N: " +  String(LoRa.packetSnr()), 5);
    sendMessage(outgoing);
    incoming = "";
  }

  display.display();
  LoRa.receive();                     // go back into receive mode
  if (millis() - lastSleepTime > sleepInterval) {
    display.clear();
    draw_msg("No response from server",0);
    draw_msg("Going to sleep",1);
    display.display();
    gotoSleep();
  }
  if(beddybyes){
    Serial.println("incoming" + incoming);
    Serial.println(String(incoming == "received"));
    if(incoming == "received"){
      display.clear();
      draw_msg("Server responded!",0);
      draw_msg("Going to sleep",1);
      display.display();
      gotoSleep();
    }
    else {
        display.clear();
        time_to_sleep = 9000;
        draw_msg("i'm drunk.....",0);
        draw_msg("Going to sleep",1);
        display.display();
        gotoSleep();  
    }
  }
}

void gotoSleep(){
  delay(2000);
//  display.sendCommand(0x8D); //into charger pump set mode
//  display.sendCommand(0x10); //turn off charger pump
//  display.sendCommand(0xAE); //set OLED sleep
  esp_deep_sleep_start();
}


void sendMessage(String message) {
  LoRa.beginPacket();                   // start packet
  LoRa.write(destination);              // add destination address
  LoRa.write(localAddress);             // add sender address
  vBat = battery_level();
  LoRa.write(vBat);
  LoRa.write(msgCount);                 // add message ID
  LoRa.write(message.length());        // add payload length
  LoRa.print(message);                 // add payload
  LoRa.endPacket();                     // finish packet and send it
  msgCount++;                           // increment message ID
}

void onReceive(int packetSize) {
  if (packetSize == 0) return;          // if there's no packet, return

  // read packet header bytes:
  recipient = LoRa.read();          // recipient address
  sender = LoRa.read();            // sender address
  incomingMsgId = LoRa.read();     // incoming msg ID
  incomingLength = LoRa.read();    // incoming msg length
  incoming = "";                 // payload of packet

  while (LoRa.available()) {            // can't use readString() in callback, so
    incoming += (char)LoRa.read();      // add bytes one by one
  }

  if (incomingLength != incoming.length()) {   // check length for error
    Serial.println("error: message length does not match length");

    return;                             // skip rest of function
  }

  // if the recipient isn't this device or broadcast,
  if (recipient == localAddress){
    beddybyes = true;
  }
  else if (recipient != 0xFF){
    Serial.println("This message is not for me.");
    return;
  }
}
