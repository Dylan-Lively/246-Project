import serial
import keyboard

ser = serial.Serial('COM3', 9600)

prev1 = None
prev2 = None
prev3 = None

while True:
    # Motor 1 - left/right (arrow keys)
    if keyboard.is_pressed('right'):
        if prev1 != 'R':
            ser.write(b'R')
            prev1 = 'R'
    elif keyboard.is_pressed('left'):
        if prev1 != 'L':
            ser.write(b'L')
            prev1 = 'L'
    else:
        if prev1 != 'S':
            ser.write(b'S')
            prev1 = 'S'

    # Motor 2 - up/down (arrow keys)
    if keyboard.is_pressed('up'):
        if prev2 != 'U':
            ser.write(b'U')
            prev2 = 'U'
    elif keyboard.is_pressed('down'):
        if prev2 != 'D':
            ser.write(b'D')
            prev2 = 'D'
    else:
        if prev2 != 'X':
            ser.write(b'X')
            prev2 = 'X'

    # Motor 3 - forward/back (W/S keys)
    if keyboard.is_pressed('w'):
        if prev3 != 'F':
            ser.write(b'F')
            prev3 = 'F'
    elif keyboard.is_pressed('s'):
        if prev3 != 'B':
            ser.write(b'B')
            prev3 = 'B'
    else:
        if prev3 != 'Z':
            ser.write(b'Z')
            prev3 = 'Z'