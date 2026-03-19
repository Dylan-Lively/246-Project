#include <AccelStepper.h>

#define STEP_PIN_1 2
#define DIR_PIN_1  5
#define STEP_PIN_2 3
#define DIR_PIN_2  6
#define STEP_PIN_3 4
#define DIR_PIN_3  7

AccelStepper stepper1(AccelStepper::DRIVER, STEP_PIN_1, DIR_PIN_1);
AccelStepper stepper2(AccelStepper::DRIVER, STEP_PIN_2, DIR_PIN_2);
AccelStepper stepper3(AccelStepper::DRIVER, STEP_PIN_3, DIR_PIN_3);

void setup() {
  Serial.begin(9600);

  stepper1.setMaxSpeed(500);
  stepper1.setAcceleration(100);

  stepper2.setMaxSpeed(500);
  stepper2.setAcceleration(100);

  stepper3.setMaxSpeed(500);
  stepper3.setAcceleration(100);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if      (cmd == 'R') stepper1.moveTo(100000);
    else if (cmd == 'L') stepper1.moveTo(-100000);
    else if (cmd == 'S') stepper1.stop();

    else if (cmd == 'U') stepper2.moveTo(100000);
    else if (cmd == 'D') stepper2.moveTo(-100000);
    else if (cmd == 'X') stepper2.stop();

    else if (cmd == 'F') stepper3.moveTo(100000);
    else if (cmd == 'B') stepper3.moveTo(-100000);
    else if (cmd == 'Z') stepper3.stop();
  }

  stepper1.run();
  stepper2.run();
  stepper3.run();
}