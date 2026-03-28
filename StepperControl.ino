/*
  3DOF Arm Controller — IK, no limit switches
  =============================================
  - Power on with arm in its neutral/home position
  - That position becomes step zero for all axes
  - Bounds enforced via step counter (Arduino) and
    normalised XYZ clamp (Python side)
  - No homing routine needed

  Serial protocol:  "X:0.623,Y:0.441,Z:0.318\n"  (normalised 0.0-1.0)
  Requires:         AccelStepper (Library Manager)
*/

#include <AccelStepper.h>
#include <math.h>

// ============================================================
// PIN ASSIGNMENTS
// ============================================================
#define BASE_STEP_PIN       4
#define BASE_DIR_PIN        7
#define SHOULDER_STEP_PIN   2
#define SHOULDER_DIR_PIN    5
#define ELBOW_STEP_PIN      3
#define ELBOW_DIR_PIN       6

// ============================================================
// PHYSICAL CONSTANTS — fill these in
// ============================================================

const float L1          = 0.24;     // shoulder axis to elbow axis (metres)
const float L2          = 0.195;    // elbow axis to tip (metres)
const float BASE_HEIGHT = 0.055;    // shoulder axis height above base origin (metres)

// Hand workspace in real metres — what volume does your hand move in?
// Laptop sends 0.0-1.0, this maps that to metres for the IK solver.
const float HAND_X_MIN = 0.0;
const float HAND_X_MAX =  1.0;
const float HAND_Y_MIN =  0.0;
const float HAND_Y_MAX =  1.0;
const float HAND_Z_MIN =  0.0;
const float HAND_Z_MAX =  1.0;

// Motor and drive constants
const int   STEPS_PER_REV       = 200;   // 200 for 1.8 degree steppers

const int   MICROSTEPS_BASE     = 1;     // match your driver MS pin settings
const int   MICROSTEPS_SHOULDER = 1;
const int   MICROSTEPS_ELBOW    = 1;

const float GEAR_RATIO_BASE     = 19.3;  // 1.0 = direct drive
const float GEAR_RATIO_SHOULDER = 18.0;
const float GEAR_RATIO_ELBOW    = 21.0;

// Zero offsets (degrees)
// When the motor is at step 0 (power-on position), what angle is
// the joint physically at in the IK reference frame?
// IK reference: shoulder 0 = horizontal, elbow 0 = fully extended
const float BASE_ZERO_OFFSET     = 0.0;
const float SHOULDER_ZERO_OFFSET = 90.0;
const float ELBOW_ZERO_OFFSET    = 0.0;

// ============================================================
// AXIS BOUNDS (steps from power-on position)
// No limit switches — you define the safe travel range manually.
// Power on with arm in neutral position, these are the max steps
// it can move in either direction from there.
// Negative = one direction, positive = other direction.
// Start conservative, widen once verified safe.
// ============================================================
const long BASE_MIN_STEPS      = -965;
const long BASE_MAX_STEPS      =  1930;

const long SHOULDER_MIN_STEPS  = -450;
const long SHOULDER_MAX_STEPS  =  450;

const long ELBOW_MIN_STEPS     = -2625;
const long ELBOW_MAX_STEPS     =  525;

// ============================================================
// MOTION PARAMETERS — start slow, increase gradually
// If a motor stalls or misses steps, reduce speed first
// ============================================================
const float BASE_MAX_SPEED      = 400.0;
const float BASE_ACCEL          = 200.0;

const float SHOULDER_MAX_SPEED  = 300.0;
const float SHOULDER_ACCEL      = 150.0;

const float ELBOW_MAX_SPEED     = 300.0;
const float ELBOW_ACCEL         = 150.0;

// ============================================================
// SERIAL
// ============================================================
#define SERIAL_BAUD         115200
#define SERIAL_TIMEOUT_MS   500     // hold position if no packet for this long

// ============================================================
// DERIVED — computed in setup() from constants above
// steps_per_degree = (STEPS_PER_REV x MICROSTEPS x GEAR_RATIO) / 360
// ============================================================
float SPD_BASE     = 0.0;
float SPD_SHOULDER = 0.0;
float SPD_ELBOW    = 0.0;

// ============================================================
// STEPPER OBJECTS
// ============================================================
AccelStepper base    (AccelStepper::DRIVER, BASE_STEP_PIN,     BASE_DIR_PIN);
AccelStepper shoulder(AccelStepper::DRIVER, SHOULDER_STEP_PIN, SHOULDER_DIR_PIN);
AccelStepper elbow   (AccelStepper::DRIVER, ELBOW_STEP_PIN,    ELBOW_DIR_PIN);

// ============================================================
// STATE
// ============================================================
enum State { STATE_RUNNING, STATE_TIMEOUT };
State currentState = STATE_RUNNING;
unsigned long lastPacketTime = 0;

char serialBuf[64];
int  serialBufIdx = 0;

// ============================================================
// UTILITIES
// ============================================================
long clampL(long val, long lo, long hi) {
  if (val < lo) return lo;
  if (val > hi) return hi;
  return val;
}

float normToReal(float norm, float minVal, float maxVal) {
  norm = constrain(norm, 0.0f, 1.0f);
  return minVal + (maxVal - minVal) * norm;
}

long angleToSteps(float angleDeg, float zeroOffsetDeg, float stepsPerDeg) {
  return (long)((angleDeg - zeroOffsetDeg) * stepsPerDeg);
}

// ============================================================
// INVERSE KINEMATICS
//
// Joint layout: base (yaw) + shoulder (pitch) + elbow (pitch)
//
// Base:
//   baseAngle = atan2(py, px)
//   — rotates to face hand in horizontal plane
//
// Collapse to 2D:
//   r = sqrt(px^2 + py^2)      horizontal distance from base axis
//   h = pz - BASE_HEIGHT        height relative to shoulder axis
//
// Elbow — law of cosines:
//   D^2 = r^2 + h^2
//   cos(elbow) = (D^2 - L1^2 - L2^2) / (2 * L1 * L2)
//   elbowAngle = acos(cos(elbow))
//
// Shoulder:
//   alpha = atan2(h, r)                               angle up to target
//   beta  = atan2(L2*sin(elbow), L1 + L2*cos(elbow)) elbow correction
//   shoulderAngle = alpha - beta
//
// Returns false if target is outside reachable workspace
// ============================================================
bool solveIK(float px, float py, float pz, float &baseAngle, float &shoulderAngle, float &elbowAngle) {

  baseAngle = atan2(py, px) * 180.0f / PI;

  float r  = sqrt(px * px + py * py);
  float h  = pz - BASE_HEIGHT;
  float D2 = r * r + h * h;
  float D  = sqrt(D2);

  // Check reachability
  if (D > (L1 + L2) || D < fabs(L1 - L2)) {
    return false;
  }

  float cosElbow = (D2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
  cosElbow   = constrain(cosElbow, -1.0f, 1.0f);
  elbowAngle = acos(cosElbow) * 180.0f / PI;

  float elbowRad = elbowAngle * PI / 180.0f;
  float alpha    = atan2(h, r);
  float beta     = atan2(L2 * sin(elbowRad), L1 + L2 * cos(elbowRad));
  shoulderAngle  = (alpha - beta) * 180.0f / PI;

  return true;
}

// ============================================================
// SERIAL — non-blocking character-by-character reader
// Accumulates into serialBuf, returns true on complete line
// ============================================================
bool readSerialLine() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      serialBuf[serialBufIdx] = '\0';
      serialBufIdx = 0;
      return true;
    }
    // Strip carriage return (handles \r\n line endings from some senders)
    if (c == '\r') continue;
    if (serialBufIdx < (int)sizeof(serialBuf) - 1)
      serialBuf[serialBufIdx++] = c;
    else
      serialBufIdx = 0;   // overflow — reset and wait for next packet
  }
  return false;
}

// FIX: sscanf with %f is unreliable on AVR. Parse manually using atof() instead.
bool parsePacket(const char* buf, float &x, float &y, float &z) {
  // Expected format: "X:0.623,Y:0.441,Z:0.318"
  const char* p = buf;

  if (p[0] != 'X' || p[1] != ':') return false;
  p += 2;
  x = atof(p);

  p = strchr(p, ',');
  if (!p) return false;
  p++;
  if (p[0] != 'Y' || p[1] != ':') return false;
  p += 2;
  y = atof(p);

  p = strchr(p, ',');
  if (!p) return false;
  p++;
  if (p[0] != 'Z' || p[1] != ':') return false;
  p += 2;
  z = atof(p);

  // Sanity check — values must be in normalised range
  if (x < 0.0f || x > 1.0f) return false;
  if (y < 0.0f || y > 1.0f) return false;
  if (z < 0.0f || z > 1.0f) return false;

  return true;
}

// ============================================================
// APPLY TARGET
// Normalised XYZ -> real metres -> IK angles -> steps -> moveTo
// ============================================================
void applyTarget(float nx, float ny, float nz) {

  float px = normToReal(nx, HAND_X_MIN, HAND_X_MAX);
  float py = normToReal(ny, HAND_Y_MIN, HAND_Y_MAX);
  float pz = normToReal(nz, HAND_Z_MIN, HAND_Z_MAX);

  float baseAng, shoulderAng, elbowAng;
  if (!solveIK(px, py, pz, baseAng, shoulderAng, elbowAng)) {
    Serial.print("IK FAILED px="); Serial.print(px, 3);
    Serial.print(" py=");           Serial.print(py, 3);
    Serial.print(" pz=");           Serial.println(pz, 3);
    return;
  }

  long baseSteps     = angleToSteps(baseAng,     BASE_ZERO_OFFSET,     SPD_BASE);
  long shoulderSteps = angleToSteps(shoulderAng, SHOULDER_ZERO_OFFSET, SPD_SHOULDER);
  long elbowSteps    = angleToSteps(elbowAng,    ELBOW_ZERO_OFFSET,    SPD_ELBOW);

  // Clamp to safe travel range — primary protection since no limit switches
  baseSteps     = clampL(baseSteps,     BASE_MIN_STEPS,     BASE_MAX_STEPS);
  shoulderSteps = clampL(shoulderSteps, SHOULDER_MIN_STEPS, SHOULDER_MAX_STEPS);
  elbowSteps    = clampL(elbowSteps,    ELBOW_MIN_STEPS,    ELBOW_MAX_STEPS);

  base.moveTo(baseSteps);
  shoulder.moveTo(shoulderSteps);
  elbow.moveTo(elbowSteps);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(SERIAL_BAUD);

  // Steps per degree for each axis
  SPD_BASE     = (STEPS_PER_REV * MICROSTEPS_BASE     * GEAR_RATIO_BASE)     / 360.0f;
  SPD_SHOULDER = (STEPS_PER_REV * MICROSTEPS_SHOULDER * GEAR_RATIO_SHOULDER) / 360.0f;
  SPD_ELBOW    = (STEPS_PER_REV * MICROSTEPS_ELBOW    * GEAR_RATIO_ELBOW)    / 360.0f;

  Serial.print("SPD base="); Serial.print(SPD_BASE);
  Serial.print(" sh=");      Serial.print(SPD_SHOULDER);
  Serial.print(" el=");      Serial.println(SPD_ELBOW);

  // All axes start at 0 — arm must be in neutral position at power-on
  base.setCurrentPosition(0);
  shoulder.setCurrentPosition(0);
  elbow.setCurrentPosition(0);

  base.setMaxSpeed(BASE_MAX_SPEED);         base.setAcceleration(BASE_ACCEL);
  shoulder.setMaxSpeed(SHOULDER_MAX_SPEED); shoulder.setAcceleration(SHOULDER_ACCEL);
  elbow.setMaxSpeed(ELBOW_MAX_SPEED);       elbow.setAcceleration(ELBOW_ACCEL);

  lastPacketTime = millis();
  Serial.println("Ready — move arm to neutral before sending commands.");
}

// ============================================================
// LOOP — no delay() anywhere, run() must execute every iteration
// ============================================================
void loop() {

  if (readSerialLine()) {
    float x, y, z;
    if (parsePacket(serialBuf, x, y, z)) {
      lastPacketTime = millis();

      if (currentState == STATE_TIMEOUT) {
        currentState = STATE_RUNNING;
        Serial.println("Signal restored.");
      }

      if (currentState == STATE_RUNNING) {
        applyTarget(x, y, z);
      }
    } else {
      Serial.print("Parse failed: [");
      Serial.print(serialBuf);
      Serial.println("]");
    }
  }

  // No packet for SERIAL_TIMEOUT_MS — stop where we are
  if (currentState == STATE_RUNNING &&
      (millis() - lastPacketTime) > SERIAL_TIMEOUT_MS) {
    currentState = STATE_TIMEOUT;
    base.moveTo(base.currentPosition());
    shoulder.moveTo(shoulder.currentPosition());
    elbow.moveTo(elbow.currentPosition());
    Serial.println("Signal lost — holding position.");
  }

  // These three lines are doing all the real work.
  // Every loop iteration each motor gets a chance to step.
  base.run();
  shoulder.run();
  elbow.run();
}