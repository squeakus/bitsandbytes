//Jonathan Byrne robo mower

#define RIGHT_FORWARD    5
#define RIGHT_BACK    3
#define LEFT_FORWARD    9
#define LEFT_BACK    6

void setup() {
  //monitor logger and motor driver
  Serial.begin(9600);
  pinMode(3, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(9, OUTPUT);
}

void motor_left(int power){
  if (power > 0){
    Serial.println("left_forward");
    analogWrite(LEFT_FORWARD, power);
    analogWrite(LEFT_BACK, 0);
  } 
  else {
    Serial.println("left_backward");
    analogWrite(LEFT_FORWARD, 0);
    analogWrite(LEFT_BACK, abs(power));
  }
}

void motor_right(int power){
  if (power > 0){
    Serial.println("right_forward");
    analogWrite(RIGHT_FORWARD, power);
    analogWrite(RIGHT_BACK, 0);
  } 
  else {
    Serial.println("right_backward");
    analogWrite(RIGHT_FORWARD, 0);
    analogWrite(RIGHT_BACK, abs(power));
  }
}

void joystick_read(){
  int x = analogRead(A0); 
  int y = analogRead(A1);  
  Serial.println("x:"+String(x)+" y:"+String(y));          // debug value
}

void loop() {
  joystick_read();
  delay(50);
//    motor_left(100);
//    delay(5000);
//    motor_left(-100);
//    delay(5000);
//    motor_right(100);
//    delay(5000);
//    motor_right(-100);
//    delay(5000);

}
