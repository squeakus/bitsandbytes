#include <SPI.h>
#include <LoRa.h>
#include<Arduino.h>
#include <SSD1306.h>


// GPIO5  -- SX1278's SCK serial clock
// GPIO19 -- SX1278's MISO
// GPIO27 -- SX1278's MOSI
// GPIO18 -- SX1278's CS chip select
// GPIO14 -- SX1278's RESET
// GPIO26 -- SX1278's IRQ(Interrupt Request)

//For the SX127x/RFM9x LoRa transceiver use:
//RST: 14, NSS: 18, SCK: 5, MOSI: 27, MISO: 19, DIO0: 26, DIO1: 33, DIO2: 32
//(DIO0 is called IRQ in several pinout diagrams and DIO1 and DIO2 are not mentioned).
//For the SSD1306 OLED display use (note: the I2C pins are non-standard):
//SCL: 15, SDA: 4, RST: 16


#define SS      18
#define RST     14
#define DI0     26
#define BAND    868E6  //915E6 -- 这里的模式选择中，检查一下是否可在中国实用915这个频段

int counter = 0;

void setup() {
  pinMode(2, OUTPUT); //Send success, LED will bright 1 second

  Serial.begin(115200);
  while (!Serial); //If just the the basic function, must connect to a computer

  SPI.begin(5, 19, 27, 18);
  LoRa.setPins(SS, RST, DI0);
  //  Serial.println("LoRa Sender");

  if (!LoRa.begin(BAND)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }
  Serial.println("LoRa Initial OK!");
}

void loop() {
  Serial.print("Sending packet: ");
  Serial.println(counter);

  // send packet
  LoRa.beginPacket();
  LoRa.print("hello ");
  LoRa.print(counter);
  LoRa.endPacket();

  counter++;
  digitalWrite(2, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);                       // wait for a second
  digitalWrite(2, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);                       // wait for a second

  delay(3000);
}
