/*
  3DOF Arm Controller
  ====================
  Starting position: shoulder straight up, forearm pointing right (L shape).
  That position is step zero and virtual tip origin.

  Coordinate convention (same on both sides, no flipping on Arduino):
      X = left/right      (positive = right)
      Y = forward/back    (positive = away from base)
      Z = up/down         (positive = up)

  Python handles the MediaPipe → arm space conversion before sending.
  Arduino just receives arm-space XYZ and uses them directly.

  Serial protocol:
      N:0.623,0.441,0.318   relative delta tracking (closed fist)
      P:0.623,0.441,0.318   absolute IK snap        (peace)
      F:0.623,0.441,0.318   hold position           (anything else)

  Requires: AccelStepper (Library Manager)
*/

#include <AccelStepper.h>
#include <math.h>

// ============================================================
// PINS
// ============================================================
#define BASE_STEP_PIN 4
#define BASE_DIR_PIN 7
#define SHOULDER_STEP_PIN 2
#define SHOULDER_DIR_PIN 5
#define ELBOW_STEP_PIN 3
#define ELBOW_DIR_PIN 6

// ============================================================
// PHYSICAL CONSTANTS
// ============================================================

// Link lengths (metres, axis to axis)
const float L1 = 0.240; // shoulder to elbow
const float L2 = 0.195; // elbow to tip

// Height of shoulder axis above base pivot (metres)
const float BASE_HEIGHT = 0.0;

// Virtual tip at L position (metres from base pivot)
// Shoulder straight up, forearm pointing right:
//   X = L2   (forearm extends along X axis to the right)
//   Y = 0    (no forward/back offset)
//   Z = L1   (upper arm goes straight up)
const float TIP_START_X = 0.195;
const float TIP_START_Y = 0.0;
const float TIP_START_Z = 0.240;

// Scale: metres of tip travel per full normalised unit (0.0 to 1.0)
// 0.2 = full hand box width moves tip 0.2m. Tune by feel.
const float SCALE_X = 0.2;
const float SCALE_Y = 0.2;
const float SCALE_Z = 0.2;

// Delta threshold — normalised units treated as stillness.
// Prevents arm tremor from hand noise. Tune 0.005 to 0.02.
const float DELTA_THRESHOLD = 0.008;

// ============================================================
// MOTOR AND DRIVE
// ============================================================
const int STEPS_PER_REV = 200;

const float GEAR_RATIO_BASE = 19.3;
const float GEAR_RATIO_SHOULDER = 18.0;
const float GEAR_RATIO_ELBOW = 21.0;

// Zero offsets (degrees)
// What angle is each joint physically at when at step 0 (L position)?
// IK reference frame: shoulder 0 = horizontal, elbow 0 = fully extended
// L position: shoulder is vertical = 90deg, elbow is 90deg bent
const float BASE_ZERO_OFFSET = 0.0;
const float SHOULDER_ZERO_OFFSET = 90.0;
const float ELBOW_ZERO_OFFSET = 90.0;

// ============================================================
// AXIS BOUNDS (steps from L position)
// ============================================================
const long BASE_MIN_STEPS = -1930;
const long BASE_MAX_STEPS = 965;

const long SHOULDER_MIN_STEPS = -300;
const long SHOULDER_MAX_STEPS = 300;

const long ELBOW_MIN_STEPS = -2450;
const long ELBOW_MAX_STEPS = 350;

// ============================================================
// MOTION PARAMETERS
// ============================================================
const float BASE_MAX_SPEED = 400.0;
const float BASE_ACCEL = 200.0;

const float SHOULDER_MAX_SPEED = 300.0;
const float SHOULDER_ACCEL = 150.0;

const float ELBOW_MAX_SPEED = 300.0;
const float ELBOW_ACCEL = 150.0;

// ============================================================
// SERIAL
// ============================================================
#define SERIAL_BAUD 115200
#define SERIAL_TIMEOUT_MS 500

// ============================================================
// DERIVED — computed in setup()
// ============================================================
float SPD_BASE = 0.0;
float SPD_SHOULDER = 0.0;
float SPD_ELBOW = 0.0;

// ============================================================
// STEPPERS
// ============================================================
AccelStepper base(AccelStepper::DRIVER, BASE_STEP_PIN, BASE_DIR_PIN);
AccelStepper shoulder(AccelStepper::DRIVER, SHOULDER_STEP_PIN, SHOULDER_DIR_PIN);
AccelStepper elbow(AccelStepper::DRIVER, ELBOW_STEP_PIN, ELBOW_DIR_PIN);

// ============================================================
// VIRTUAL TIP
// Arm's believed tip position in arm space (metres).
// Starts at L position. All modes update this.
// ============================================================
float tipX = TIP_START_X;
float tipY = TIP_START_Y;
float tipZ = TIP_START_Z;

// Previous normalised hand coords for delta calculation
bool prevInitialised = false;
float prevNX = 0.5;
float prevNY = 0.5;
float prevNZ = 0.5;

// ============================================================
// STATE
// ============================================================
enum State
{
    STATE_RUNNING,
    STATE_TIMEOUT
};
State currentState = STATE_RUNNING;
unsigned long lastPacketTime = 0;

char serialBuf[64];
int serialBufIdx = 0;

// ============================================================
// UTILITIES
// ============================================================
long clampL(long val, long lo, long hi)
{
    if (val < lo)
        return lo;
    if (val > hi)
        return hi;
    return val;
}

float clampF(float val, float lo, float hi)
{
    if (val < lo)
        return lo;
    if (val > hi)
        return hi;
    return val;
}

long angleToSteps(float angleDeg, float zeroOffsetDeg, float stepsPerDeg)
{
    return (long)((angleDeg - zeroOffsetDeg) * stepsPerDeg);
}

// ============================================================
// INVERSE KINEMATICS
//
// All inputs and outputs are in consistent arm space:
//   X = left/right, Y = forward/back, Z = up/down
//
// Base:
//   Rotates to face target in horizontal (XY) plane.
//   atan2(px, py) — base homes facing down Y, X is lateral deviation.
//   If base homes facing right (down X), swap to atan2(py, px).
//
// Collapse to 2D vertical plane:
//   r = sqrt(px^2 + py^2)   horizontal distance from base axis
//   h = pz - BASE_HEIGHT     height relative to shoulder axis
//
// Elbow via law of cosines:
//   D^2 = r^2 + h^2
//   cos(elbow) = (D^2 - L1^2 - L2^2) / (2 * L1 * L2)
//   elbow = acos(cos(elbow))
//
// Shoulder:
//   alpha = atan2(h, r)
//   beta  = atan2(L2 * sin(elbow), L1 + L2 * cos(elbow))
//   shoulder = alpha - beta
//
// Returns false if target is outside reachable workspace
// ============================================================
bool solveIK(float px, float py, float pz,
             float &baseAngle, float &shoulderAngle, float &elbowAngle)
{

    // Base homes facing right (down X axis) since forearm starts pointing right
    // so lateral deviation is Y, and atan2(py, px) gives correct rotation
    baseAngle = atan2(py, px) * 180.0f / PI;

    float r = sqrt(px * px + py * py);
    float h = pz - BASE_HEIGHT;
    float D2 = r * r + h * h;
    float D = sqrt(D2);

    if (D > (L1 + L2) || D < fabs(L1 - L2))
    {
        return false;
    }

    float cosElbow = (D2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
    cosElbow = constrain(cosElbow, -1.0f, 1.0f);
    elbowAngle = acos(cosElbow) * 180.0f / PI;

    float elbowRad = elbowAngle * PI / 180.0f;
    float alpha = atan2(h, r);
    float beta = atan2(L2 * sin(elbowRad), L1 + L2 * cos(elbowRad));
    shoulderAngle = (alpha - beta) * 180.0f / PI;

    return true;
}

// ============================================================
// MOVE TO XYZ
// All modes call this. Takes a target in arm space (metres),
// runs IK, converts to steps, clamps, commands motors.
// ============================================================
void moveToXYZ(float px, float py, float pz)
{
    float baseAng, shoulderAng, elbowAng;

    if (!solveIK(px, py, pz, baseAng, shoulderAng, elbowAng))
    {
        Serial.print("IK unreachable: ");
        Serial.print(px, 3);
        Serial.print(" ");
        Serial.print(py, 3);
        Serial.print(" ");
        Serial.println(pz, 3);
        return;
    }

    long baseSteps = angleToSteps(baseAng, BASE_ZERO_OFFSET, SPD_BASE);
    long shoulderSteps = angleToSteps(shoulderAng, SHOULDER_ZERO_OFFSET, SPD_SHOULDER);
    long elbowSteps = angleToSteps(elbowAng, ELBOW_ZERO_OFFSET, SPD_ELBOW);

    baseSteps = clampL(baseSteps, BASE_MIN_STEPS, BASE_MAX_STEPS);
    shoulderSteps = clampL(shoulderSteps, SHOULDER_MIN_STEPS, SHOULDER_MAX_STEPS);
    elbowSteps = clampL(elbowSteps, ELBOW_MIN_STEPS, ELBOW_MAX_STEPS);

    base.moveTo(baseSteps);
    shoulder.moveTo(shoulderSteps);
    elbow.moveTo(elbowSteps);

    // Debug — comment out once verified
    Serial.print("ang ");
    Serial.print(baseAng, 1);
    Serial.print(" ");
    Serial.print(shoulderAng, 1);
    Serial.print(" ");
    Serial.println(elbowAng, 1);
    Serial.print("stp ");
    Serial.print(baseSteps);
    Serial.print(" ");
    Serial.print(shoulderSteps);
    Serial.print(" ");
    Serial.println(elbowSteps);
}

// ============================================================
// MODE HANDLERS
// ============================================================

// N — relative delta tracking
// Accumulates scaled deltas into virtual tip, calls moveToXYZ.
// No axis flipping here — Python already sent arm-space coords.
void handleRelative(float nx, float ny, float nz)
{

    if (!prevInitialised)
    {
        prevNX = nx;
        prevNY = ny;
        prevNZ = nz;
        prevInitialised = true;
        return;
    }

    float dx = nx - prevNX;
    float dy = ny - prevNY;
    float dz = nz - prevNZ;

    // Scale normalised deltas to metres
    float armDX = dx * SCALE_X;
    float armDY = dy * SCALE_Y;
    float armDZ = dz * SCALE_Z;

    // Deadband — ignore sub-threshold movement
    if (fabs(armDX) < DELTA_THRESHOLD * SCALE_X)
        armDX = 0.0f;
    if (fabs(armDY) < DELTA_THRESHOLD * SCALE_Y)
        armDY = 0.0f;
    if (fabs(armDZ) < DELTA_THRESHOLD * SCALE_Z)
        armDZ = 0.0f;

    // Accumulate into virtual tip
    tipX += armDX;
    tipY += armDY;
    tipZ += armDZ;

    // Clamp virtual tip to reachable envelope
    float reach = (L1 + L2) * 0.9f;
    tipX = clampF(tipX, -reach, reach);
    tipY = clampF(tipY, 0.01f, reach);
    tipZ = clampF(tipZ, BASE_HEIGHT, L1 + L2 + BASE_HEIGHT);

    moveToXYZ(tipX, tipY, tipZ);

    // Always update prev — even if delta was zeroed by deadband
    prevNX = nx;
    prevNY = ny;
    prevNZ = nz;
}

// F — hold position
// Do not update prev — when tracking resumes, first delta
// is calculated from current hand position correctly.
void handleHold()
{
    prevInitialised = false;
    base.moveTo(base.currentPosition());
    shoulder.moveTo(shoulder.currentPosition());
    elbow.moveTo(elbow.currentPosition());
}

// P — absolute IK snap
// Maps hand position directly to arm space.
// Centre of hand box (0.5) maps to arm L position (TIP_START).
// Syncs virtual tip and prev so delta resumes correctly after snap.
void handleAbsolute(float nx, float ny, float nz)
{

    float armX = TIP_START_X + (nx - 0.5f) * SCALE_X;
    float armY = TIP_START_Y + (ny - 0.5f) * SCALE_Y;
    float armZ = TIP_START_Z + (nz - 0.5f) * SCALE_Z;

    moveToXYZ(armX, armY, armZ);

    // Sync virtual tip and prev so delta resumes from here
    tipX = armX;
    tipY = armY;
    tipZ = armZ;

    prevNX = nx;
    prevNY = ny;
    prevNZ = nz;
}

// ============================================================
// SERIAL
// ============================================================
bool readSerialLine()
{
    while (Serial.available() > 0)
    {
        char c = Serial.read();
        if (c == '\n')
        {
            serialBuf[serialBufIdx] = '\0';
            serialBufIdx = 0;
            return true;
        }
        if (c == '\r')
            continue;
        if (serialBufIdx < (int)sizeof(serialBuf) - 1)
            serialBuf[serialBufIdx++] = c;
        else
            serialBufIdx = 0;
    }
    return false;
}

// Parse "N:0.623,0.441,0.318"
bool parsePacket(const char *buf, char &prefix, float &x, float &y, float &z)
{
    if (strlen(buf) < 3)
        return false;
    if (buf[1] != ':')
        return false;

    prefix = buf[0];

    const char *p = buf + 2;
    x = atof(p);
    p = strchr(p, ',');
    if (!p)
        return false;
    p++;
    y = atof(p);
    p = strchr(p, ',');
    if (!p)
        return false;
    p++;
    z = atof(p);

    if (x < 0.0f || x > 1.0f)
        return false;
    if (y < 0.0f || y > 1.0f)
        return false;
    if (z < 0.0f || z > 1.0f)
        return false;

    return true;
}

// ============================================================
// SETUP
// ============================================================
void setup()
{
    Serial.begin(SERIAL_BAUD);

    SPD_BASE = (STEPS_PER_REV * GEAR_RATIO_BASE) / 360.0f;
    SPD_SHOULDER = (STEPS_PER_REV * GEAR_RATIO_SHOULDER) / 360.0f;
    SPD_ELBOW = (STEPS_PER_REV * GEAR_RATIO_ELBOW) / 360.0f;

    Serial.print("SPD base=");
    Serial.print(SPD_BASE, 3);
    Serial.print(" sh=");
    Serial.print(SPD_SHOULDER, 3);
    Serial.print(" el=");
    Serial.println(SPD_ELBOW, 3);

    Serial.print("TIP start X=");
    Serial.print(TIP_START_X, 3);
    Serial.print(" Y=");
    Serial.print(TIP_START_Y, 3);
    Serial.print(" Z=");
    Serial.println(TIP_START_Z, 3);

    base.setCurrentPosition(0);
    shoulder.setCurrentPosition(0);
    elbow.setCurrentPosition(0);

    base.setMaxSpeed(BASE_MAX_SPEED);
    base.setAcceleration(BASE_ACCEL);
    shoulder.setMaxSpeed(SHOULDER_MAX_SPEED);
    shoulder.setAcceleration(SHOULDER_ACCEL);
    elbow.setMaxSpeed(ELBOW_MAX_SPEED);
    elbow.setAcceleration(ELBOW_ACCEL);

    lastPacketTime = millis();
    Serial.println("Ready. Arm must be in L position at power on.");
}

// ============================================================
// LOOP — no delay() anywhere
// ============================================================
void loop()
{

    if (readSerialLine())
    {
        char prefix;
        float x, y, z;

        if (parsePacket(serialBuf, prefix, x, y, z))
        {
            lastPacketTime = millis();

            if (currentState == STATE_TIMEOUT)
            {
                currentState = STATE_RUNNING;
                Serial.println("Signal restored.");
            }

            if (currentState == STATE_RUNNING)
            {
                switch (prefix)
                {
                case 'N':
                    handleRelative(x, y, z);
                    break;
                case 'F':
                    handleHold();
                    break;
                case 'P':
                    handleAbsolute(x, y, z);
                    break;
                default:
                    Serial.print("Unknown prefix: ");
                    Serial.println(prefix);
                    break;
                }
            }
        }
        else
        {
            Serial.print("Parse failed: [");
            Serial.print(serialBuf);
            Serial.println("]");
        }
    }

    if (currentState == STATE_RUNNING &&
        (millis() - lastPacketTime) > SERIAL_TIMEOUT_MS)
    {
        currentState = STATE_TIMEOUT;
        base.moveTo(base.currentPosition());
        shoulder.moveTo(shoulder.currentPosition());
        elbow.moveTo(elbow.currentPosition());
        Serial.println("Signal lost — holding.");
    }

    base.run();
    shoulder.run();
    elbow.run();
}
