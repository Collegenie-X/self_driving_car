import cv2
import numpy as np
import YB_Pcb_Car
import threading
import time
import RPi.GPIO as GPIO

# Camera and car initialization
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Set width
cap.set(4, 240)  # Set height
cap.set(cv2.CAP_PROP_BRIGHTNESS, 70)
cap.set(cv2.CAP_PROP_CONTRAST, 60)
cap.set(cv2.CAP_PROP_SATURATION, 40)
cap.set(cv2.CAP_PROP_GAIN, 40)

car = YB_Pcb_Car.YB_Pcb_Car()

# Constants
MOTOR_UP_SPEED = 115    # Speed range: 65 ~ 125
MOTOR_DOWN_SPEED = 70
DETECT_VALUE = 30        # Brightness value range: 70/130

# Haar Cascade models
traffic_light_cascade = cv2.CascadeClassifier('3_traffic_sign_cascade.xml')




# 이미지의 평균 밝기 계산 함수
def calculate_average_brightness(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(grayscale_image)

# 자동 밝기 및 대비 조절 함수
def auto_adjust_brightness_contrast(image, target_brightness=128, target_contrast=100):
    current_brightness = calculate_average_brightness(image)
    
    # 밝기 조절 계수 계산
    brightness_difference = target_brightness - current_brightness
    brightness_factor = brightness_difference / 255.0
    adjusted_image = cv2.addWeighted(image, 1 + brightness_factor, image, 0, brightness_difference)
    
    # 대비 조절 계수 계산
    contrast_factor = 131 * (target_contrast + 127) / (127 * (131 - target_contrast))
    adjusted_image = cv2.addWeighted(adjusted_image, contrast_factor, adjusted_image, 0, 127 * (1 - contrast_factor))
    
    return adjusted_image


def detect_traffic_light(frame, control_signals):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    traffic_lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if not len(traffic_lights) : 
        return 
    for (x, y, w, h) in traffic_lights:
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
       
    cv2.putText(img,"traffic_light", (x-30,y+20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0))
    cv2.imshow("traffic_light",img)
    control_signals['red_light'] = len(traffic_lights) > 0



# Autonomous driving functions
def process_frame(frame):
    pts_src = np.float32([[10, 80], [310, 80], [310, 10], [10, 10]])
    pts_dst = np.float32([[0, 240], [320, 240], [320, 0], [0, 0]])

    # 사각형 그리기
    pts = pts_src.reshape((-1, 1, 2)).astype(np.int32)  # np.float32에서 np.int32로 변경
    frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imshow('1_Frame', frame)

    # Apply perspective transformation
    mat_affine = cv2.getPerspectiveTransform(pts_src, pts_dst)
    frame_transformed = cv2.warpPerspective(frame, mat_affine, (320, 240))
    cv2.imshow('2_frame_transformed', frame_transformed)

    # Convert to grayscale and apply binary threshold
    gray_frame = cv2.cvtColor(frame_transformed, cv2.COLOR_RGB2GRAY)
    cv2.imshow('3_gray_frame', gray_frame)
    _, binary_frame = cv2.threshold(gray_frame, DETECT_VALUE, 255, cv2.THRESH_BINARY)
    return binary_frame

def decide_direction(roi):
    left = int(np.sum(histogram[:int(len(histogram) / 4)]))
    right = int(np.sum(histogram[int(3 * len(histogram) / 4):]))
    up = np.sum(histogram[int(len(histogram) / 4):int(3 * len(histogram) / 4)])

    print("left:", left)
    print("right:", right)
    print("up:", up)

    if abs(right - left) > 1000:
        return "LEFT" if right > left else "RIGHT"
    elif up < 10000:
        return "STRAIGHT"
    else:
        return "STRAIGHT"

def control_car(direction):
    print(f"Controlling car: {direction}")
    if direction == "STRAIGHT":
        car.Car_Run(MOTOR_UP_SPEED - 35, MOTOR_UP_SPEED - 35)
    elif direction == "LEFT":
        car.Car_Left(MOTOR_DOWN_SPEED, MOTOR_UP_SPEED)
    elif direction == "RIGHT":
        car.Car_Right(MOTOR_UP_SPEED, MOTOR_DOWN_SPEED)
    elif direction == "RANDOM":
        random_direction = random.choice(["LEFT", "RIGHT"])
        control_car(random_direction)   

def rotate_servo(servo_id, angle):
    car.Ctrl_Servo(servo_id, angle)    



def finish_exit() : 
    car.Car_Stop()
    cap.release()
    cv2.destroyAllWindows()        


    

# Main loop
try:
    rotate_servo(1, 90)  # Rotate servo at S1 to 90 degrees
    rotate_servo(2, 110)  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        cv2.imshow('0_Frame', frame)
        # frame = auto_adjust_brightness_contrast(frame,120,120)


        # Shared control signals dictionary
        control_signals = {'red_light': False}

        

        # Create and start threads for detection tasks

        try : 
            traffic_light_thread = threading.Thread(target=detect_traffic_light, args=(frame.copy(), control_signals))
            traffic_light_thread.start()
           
        except Exception as E: 
            print("###############")
            print("###### error: ",E)
            print("################")
            car.Car_Stop() 
            time.sleep(2)

        # Wait for threads to finish
        traffic_light_thread.join()
     
        # Autonomous driving logic based on detections
     
         
        if control_signals['red_light']:
            print("Red light detected! Stopping...")            
            car.Car_Stop()  # Stop the car
            time.sleep(1)
    
        else:
            roi = process_frame(frame)
            histogram = np.sum(roi, axis=0)
            direction = decide_direction(histogram)
            control_car(direction)

            print(f"Histogram: {histogram}")
            print(f"Decided direction: {direction}")
                    

        # Pause/Unpause and Exit logic
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space bar to pause/unpause
            cv2.waitKey(0)  # Wait until any key is pressed
        elif key == 27:  # ESC to quit
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    finish_exit()
    
