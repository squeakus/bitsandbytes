int pinCount = 18;
//int ADCpins[] = {2,12,13,32,33,34,35,36,37,38,39}; // these are the ADC pins

int ADCpins[] = {0, 2, 4, 12, 13, 14, 15, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39}; // these are the WEMOS ADC pins

float VBAT;  // battery voltage from ESP32 ADC read float
float ADC_divider = 4.2;  // voltage divider proportions - hypothetical so far :-)

void setup() {
  Serial.begin(9600);                   // initialize serial
  while (!Serial);

  for (int thisPin = 0; thisPin < pinCount; thisPin++) {

    Serial.print(thisPin, DEC);     Serial.print(" = ");     Serial.print(ADCpins[thisPin], DEC);     Serial.print(" => ");

    pinMode(ADCpins[thisPin], INPUT);
    float rawvalue = (float)analogRead(ADCpins[thisPin]);
    VBAT = (ADC_divider * rawvalue) / 4096.0; // LiPo battery voltage in volts
    Serial.print("Vbat = "); Serial.print(VBAT); Serial.print(" Volts Raw:"); Serial.println(rawvalue);
  }
}

void loop() {

}
