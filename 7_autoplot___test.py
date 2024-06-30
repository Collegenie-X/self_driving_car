import cv2
import numpy as np
import YB_Pcb_Car
import random
import time

# Camera and car initialization
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # set Width
cap.set(4, 240)  # set Height

car = YB_Pcb_Car.YB_Pcb_Car()

# Constants 초기값
MOTOR_UP_SPEED = 115    ####   65 ~ 125 Speed 
MOTOR_DOWN_SPEED = 70   
 #### 밝기 70/130 

# 트랙바 콜백 함수 (사용되지 않음)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Camera Settings')

# 트랙바 생성
cv2.createTrackbar('Brightness', 'Camera Settings', 40, 100, nothing)
cv2.createTrackbar('Contrast', 'Camera Settings', 40, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Detect Value', 'Camera Settings', 30, 120, nothing)
cv2.createTrackbar('Motor Up Speed', 'Camera Settings', 90, 125, nothing)
cv2.createTrackbar('Motor Down Speed', 'Camera Settings', 50, 125, nothing)

def process_frame(frame, detect_value):
    """
    Process the frame to detect edges and transform perspective.
    """
    # Define region for perspective transformation
    
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
    _, binary_frame = cv2.threshold(gray_frame, detect_value, 255, cv2.THRESH_BINARY)
    return binary_frame

def decide_direction(histogram):
    """
    Decide the driving direction based on histogram.
    """
    left = int(np.sum(histogram[:int(len(histogram) / 4)]))
    right = int(np.sum(histogram[int(3 * len(histogram) / 4):]))
    up = np.sum(histogram[int(len(histogram) / 4):int(3 * len(histogram) / 4)])

    print("left:", left)
    print("right:", right)
    print("up:", up)

    if abs(right - left) > 1000:
        return "LEFT" if right > left else "RIGHT"
    elif up < 10000:
        return "UP"
    else:
        return "UP"

def control_car(direction, up_speed, down_speed):
    """
    Control the car based on the decided direction.
    """
    print(f"Controlling car: {direction}")
    if direction == "UP":
        car.Car_Run(up_speed - 35, up_speed - 35)
    elif direction == "LEFT":
        car.Car_Left(down_speed, up_speed)
    elif direction == "RIGHT":
        car.Car_Right(up_speed, down_speed)
    elif direction == "RANDOM":
        random_direction = random.choice(["LEFT", "RIGHT"])
        control_car(random_direction, up_speed, down_speed)    

def rotate_servo(servo_id, angle):
    car.Ctrl_Servo(servo_id, angle)    

try:
    rotate_servo(1, 90)  # Rotate servo at S1 to 90 degrees
    rotate_servo(2, 110)  

    while True:
        # 트랙바 값 읽기
        brightness = cv2.getTrackbarPos('Brightness', 'Camera Settings')
        contrast = cv2.getTrackbarPos('Contrast', 'Camera Settings')
        saturation = cv2.getTrackbarPos('Saturation', 'Camera Settings')
        gain = cv2.getTrackbarPos('Gain', 'Camera Settings')
        detect_value = cv2.getTrackbarPos('Detect Value', 'Camera Settings')
        motor_up_speed = cv2.getTrackbarPos('Motor Up Speed', 'Camera Settings')
        motor_down_speed = cv2.getTrackbarPos('Motor Down Speed', 'Camera Settings')

        # 카메라 속성 설정
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        
        processed_frame = process_frame(frame, detect_value)
        histogram = np.sum(processed_frame, axis=0)
        print(f"Histogram: {histogram}")
        direction = decide_direction(histogram)
        print(f"Decided direction: {direction}")
        control_car(direction, motor_up_speed, motor_down_speed)

        # Display the processed frame (for debugging)
        cv2.imshow('4_Processed Frame', processed_frame)
        

        key = cv2.waitKey(30) & 0xff
        if key == 27:  # press 'ESC' to quit
            break
        elif key == 32:  # press 'Space bar' for pause and debug
            print("Paused for debugging. Press any key to continue.")
            cv2.waitKey()

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    car.Car_Stop()
    cap.release()
    cv2.destroyAllWindows()
