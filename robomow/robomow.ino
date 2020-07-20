//Jonathan Byrne robo mower

#define LEFT_FORWARD    5
#define LEFT_BACK    3
#define RIGHT_FORWARD    9
#define RIGHT_BACK    6

void setup() {
  //monitor logger and motor driver
  Serial.begin(9600);
  pinMode(3, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(9, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:

}
