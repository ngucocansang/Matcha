import serial
import time
import math

ser = serial.Serial("COM5", 9600) #check COM nha 
time.sleep(2)

start_time = time.time()

while time.time() - start_time < 15:  # 15 giây
    line = ser.readline().decode().strip()
    try:
        pitch_deg, pitch_rate_deg = map(float, line.split(","))
        pitch = math.radians(pitch_deg)
        pitch_rate = math.radians(pitch_rate_deg)

        # PD controller
        action = -pitch * 15 - pitch_rate * 2

        ser.write(f"{action}\n".encode())

        print(f"Pitch: {pitch_deg:.2f}, Rate: {pitch_rate_deg:.2f}, Action: {action:.2f}")
    except:
        continue

ser.close()
