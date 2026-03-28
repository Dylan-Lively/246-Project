import serial
import time

ser = serial.Serial('COM4', 115200)
time.sleep(3)

while True:
    ser.write(b'HELLO\n')
    print("Sent HELLO")
    time.sleep(1)