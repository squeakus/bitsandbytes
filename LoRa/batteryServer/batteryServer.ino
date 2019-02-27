/*
  LoRa Battery Server

  connects to the wifi
  Sends a message every half second, and uses callback
  for new incoming messages. Implements a one-byte addressing scheme,
  with 0xFF as the broadcast address.

  Note: while sending, LoRa radio is not listening for incoming messages.
  Note2: when using the callback method, you can't use any of the Stream
  functions that rely on the timeout, such as readString, parseInt(), etc.

  created 28 April 2017
  based on code by Tom Igoe
*/
#include <LoRa.h>
#include <WiFi.h>
#include<Arduino.h>
#include "SSD1306.h"
#include "ThingSpeak.h"
#include "time.h"
#include "secrets.h"
#define SCK 5 // GPIO5 - SX1278's SCK serial clock
#define MISO 19 // GPIO19 - SX1278's MISO
#define MOSI 27 // GPIO27 - SX1278's MOSI
#define CS 18 // GPIO18 - SX1278's CS Radio chip select
#define RST 14 // GPIO14 - SX1278's LoRa radio reset
#define DI0 26 // GPIO26 - SX1278's IRQ (interrupt request) change for your board; must be a hardware interrupt pin
#define BAND 868E6 // Frequencxy band to operate on.

String outgoing;              // outgoing message
byte msgCount = 0;            // count of outgoing messages
byte localAddress = 0xAA;     // address of this device
byte destination = 0xFF;
long lastSendTime = 0;        // last send time
int interval = 2000;          // interval between sends
int recipient = 0;          // recipient address
byte sender;            // sender address
byte incomingMsgId;     // incoming msg ID
byte incomingLength;    // incoming msg length
String incoming = "";                 // payload of packet
String lastMessage = "";
byte vByte = 0;
float vBat = 0;
const char* ssid       = SECRET_SSID;
const char* password   = SECRET_PASS;
unsigned long myChannelNumber = SECRET_CH_ID;
const char * myWriteAPIKey = SECRET_WRITE_APIKEY;
WiFiClient  client;

SSD1306 display (0x3c, 4, 15); // OLED Screen
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 3600;
const int   daylightOffset_sec = 3600;

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
  draw_msg("LoRa Battery Server", 0);
  display.display();
  delay(1500);
}

void thing_update(float value){
  // Write value to Field 1 of a ThingSpeak Channel
  Serial.println("Channel write starting.");
  int httpCode = ThingSpeak.writeField(myChannelNumber, 2, value, myWriteAPIKey);

  if (httpCode == 200) {
    Serial.println("Channel write successful.");
  }
  else {
    Serial.println("Problem writing to channel. HTTP error code " + String(httpCode));
  }
}

void draw_msg(String msg, int line) {
  int spacing = line * 10;
  display.drawString (0, spacing, msg);
}

String get_time() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return "";
  }
  char timeStringBuff[50]; //50 chars should be enough
  strftime(timeStringBuff, sizeof(timeStringBuff), " %d/%m/%y %H:%M:%S", &timeinfo);
  //print like "const char*"
  Serial.println(timeStringBuff);
  //Optional: Construct String object
  String curTime(timeStringBuff);
  return curTime;
}

void update_info(String message){
  display.clear();
  draw_msg("Server: " + String(localAddress, HEX) + " To: " + String(destination, HEX), 0);
  draw_msg("Msg:" + message, 1);
  if (lastMessage != "") {
    vBat= 4.2 * (float(vByte) / 255);
    draw_msg("From: " + String(sender, HEX) + " MsgID: " + String(incomingMsgId), 3);
    draw_msg("Msg:" + lastMessage, 4);
    draw_msg("Battery: " + String(vBat) + "byte: "+ String(vByte), 5);
  }
  else {
    draw_msg("From: ", 3);
    draw_msg("Msg:", 4);
    draw_msg("Battery:", 5);
  }
  display.display();
}

void setup() {
  Serial.begin(9600);                   // initialize serial
  while (!Serial);

  init_display();

  // connect to wifi and set up time server
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    draw_msg("Connecting: " + String(ssid), 2);
    display.display();
    delay(500);
    Serial.print(".");
  }


  //init and get the time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  // Turn on thingspeak
  WiFi.mode(WIFI_STA);
  ThingSpeak.begin(client);

  // override the default CS, reset, and IRQ pins (optional)
  LoRa.setPins(CS, RST, DI0);// set CS, reset, IRQ pin

  if (!LoRa.begin(BAND)) {             // initialize ratio at 868 MHz
    Serial.println("LoRa init failed. Check your connections.");
    while (true);                       // if failed, do nothing
  }

  LoRa.onReceive(onReceive);
  LoRa.receive();

  Serial.println("LoRa init succeeded.");
}

void loop() {
  // check if we have received a message
  if (incoming != ""){
    String out = "received:"  + get_time();
    update_info(out);
    sendMessage(out, destination);
    thing_update(vBat);
    lastMessage = incoming;
    incoming = "";
  }

  // Send a message out
  if (millis() - lastSendTime > interval) {
    destination = 0xFF; // reset to broadcast
    String message = get_time();
    sendMessage(message, destination);
    update_info(message);
    Serial.println("Sending " + message);
    lastSendTime = millis();            // timestamp the message
    LoRa.receive();                     // go back into receive mode
  }

}

void sendMessage(String outgoing, byte destination) {
  LoRa.beginPacket();                   // start packet
  LoRa.write(destination);              // add destination address
  LoRa.write(localAddress);             // add sender address
  LoRa.write(msgCount);                 // add message ID
  LoRa.write(outgoing.length());        // add payload length
  LoRa.print(outgoing);                 // add payload
  LoRa.endPacket();                     // finish packet and send it
  msgCount++;                           // increment message ID
}

void onReceive(int packetSize) {
  if (packetSize == 0) return;          // if there's no packet, return

  // read packet header bytes:
  recipient = LoRa.read();         // recipient address
  sender = LoRa.read();            // sender address
  vByte = LoRa.read();             // Byte representation of battery level
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
  if (recipient != localAddress && recipient != 0xFF) {
    Serial.println("This message is not for me.");
    return;                             // skip rest of function
  }
  
  destination = sender;
}
