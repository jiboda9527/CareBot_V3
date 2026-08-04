import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

sys.path.append(
    "/home/pi/project_demo/07.AI_Visual_Recognition/04.Tensorflow_object_recognition"
)

#导入Raspbot驱动库 Import the Raspbot library
from McLumk_Wheel_Sports import *
# 创建Rosmaster对象 Facebot Create the Rosmaster object Facebot
Facebot = Raspbot()

import cv2
import mediapipe as mp
import PID
import tensorflow as tf
from object_detection.utils import label_map_util

import numpy as np
from yolo_person_detector import YoloPersonDetector


direction_pid = PID.PositionalPID(0.2, 0, 0.002)
speed_pid = PID.PositionalPID(0.01, 0,0.0001)
area_center = 180000 #面积大小
MIN_Speed = 2
MAX_MOTOR_SPEED = 255
PAN_CENTER = 90
PAN_MIN = 20
PAN_MAX = 160
PAN_DEAD_ZONE = 20
PAN_GAIN = 0.02
PAN_MAX_STEP = 2.0
PAN_CHASSIS_MARGIN = 12
TILT_CENTER = 50
TILT_MIN = 5
TILT_MAX = 95
TILT_DEAD_ZONE = 18
TILT_GAIN = 0.025
TILT_MAX_STEP = 3.0

# ===========================
# Fall Detection Parameters
# ===========================

# 宽高比阈值（w/h）
FALL_RATIO_THRESHOLD = 1.2

# 连续多少帧判定跌倒（20FPS时60帧≈3秒）
FALL_CONFIRM_FRAMES = 60
FALL_DEBUG = False

# 跌倒计数器
fall_frame_count = 0

# 当前跌倒状态
is_fall = False


# 控制电机运动 Control motor movement
def run_motor(M1,M2,M3,M4):  #-255~255
    motor_values = [
        limit_max_vlaue(int(M1), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED),
        limit_max_vlaue(int(M2), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED),
        limit_max_vlaue(int(M3), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED),
        limit_max_vlaue(int(M4), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED),
    ]
    Facebot.Ctrl_Muto(0, motor_values[0])
    Facebot.Ctrl_Muto(1, motor_values[1])
    Facebot.Ctrl_Muto(2, motor_values[2])
    Facebot.Ctrl_Muto(3, motor_values[3])
    
def limit_max_vlaue(a,min,max):
    if a<min:
        a=min
    if a>max:
        a=max
    return a

def limit_speed(a,min):#限制最小速度
    if a <0:
        if a> -min:
            a = -min
    elif a>0:
        if a<min:
            a = min
    return a


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

#传入参数 x 和 y轴
def control_motor_speed(speed_fb,speed_lr):
    speed_L1 = speed_fb + speed_lr 
    speed_L2 = speed_fb + speed_lr 
    speed_R1 = speed_fb - speed_lr 
    speed_R2 = speed_fb - speed_lr 
    #满足速度范围
    speed_L1 = limit_speed(speed_L1,MIN_Speed)
    speed_L2 = limit_speed(speed_L2,MIN_Speed)
    speed_R1 = limit_speed(speed_R1,MIN_Speed)
    speed_R2 = limit_speed(speed_R2,MIN_Speed)

    #print(speed_L1,speed_L2,speed_R1,speed_R2)
    run_motor(speed_L1,speed_L2,speed_R1,speed_R2) #控制电机转


def reset_pid(pid):
    pid.SystemOutput = 0.0
    pid.ResultValueBack = 0.0
    pid.PidOutput = 0.0
    pid.PIDErrADD = 0.0
    pid.ErrBack = 0.0


def reset_tracking_pid():
    reset_pid(direction_pid)
    reset_pid(speed_pid)

def fall_detect(w, h):
    """
    第一版跌倒检测

    条件：
    1. 宽高比(w/h)大于阈值
    2. 连续保持一定帧数

    返回：
        True  -> 跌倒
        False -> 正常
    """

    global fall_frame_count
    global is_fall

    # 防止除0
    if h <= 0:
        return False

    ratio = w / float(h)

    # 调试信息
    if FALL_DEBUG:
        print(f"ratio={ratio:.2f}")

    # 判断人体是否接近水平
    if ratio > FALL_RATIO_THRESHOLD:
        fall_frame_count += 1
    else:
        fall_frame_count = 0
        is_fall = False

    if FALL_DEBUG:
        print(f"fall_count={fall_frame_count}")

    # 连续达到指定帧数
    if fall_frame_count >= FALL_CONFIRM_FRAMES:
        is_fall = True

    return is_fall

class PersonDetector:

    def __init__(self):

        MODEL_NAME = '/home/pi/project_demo/07.AI_Visual_Recognition/04.Tensorflow_object_recognition/ssdlite_mobilenet_v2_coco_2018_05_09'

        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        PATH_TO_LABELS = '/home/pi/project_demo/07.AI_Visual_Recognition/04.Tensorflow_object_recognition/data/mscoco_label_map.pbtxt'

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():

            od_graph_def = tf.compat.v1.GraphDef()

            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

                serialized_graph = fid.read()

                od_graph_def.ParseFromString(serialized_graph)

                tf.import_graph_def(
                    od_graph_def,
                    name=''
                )

        label_map = label_map_util.load_labelmap(
            PATH_TO_LABELS
        )

        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=90,
            use_display_name=True
        )

        self.category_index = (
            label_map_util.create_category_index(
                categories
            )
        )

        self.sess = tf.compat.v1.Session(
            graph=self.detection_graph
        )
        self.last_inference_ms = 0.0

        print("SSD Person Detector Loaded")


    def findPersons(self, frame):

        bboxs = []
        bbox = (0,0,0,0)
        center_x = 0

        image_np_expanded = np.expand_dims(
            frame,
            axis=0
        )

        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0'
        )

        detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0'
        )

        detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0'
        )

        detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0'
        )

        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0'
        )

        inference_started = time.perf_counter()
        boxes, scores, classes, num = self.sess.run(
            [
                detection_boxes,
                detection_scores,
                detection_classes,
                num_detections
            ],
            feed_dict={
                image_tensor:image_np_expanded
            }
        )
        self.last_inference_ms = (time.perf_counter() - inference_started) * 1000.0

        h_img, w_img, _ = frame.shape

        largest_area = 0

        for i in range(10):

            if scores[0][i] < 0.5:
                continue

            cls_name = self.category_index[
                int(classes[0][i])
            ]['name']

            if cls_name != "person":
                continue

            ymin, xmin, ymax, xmax = boxes[0][i]

            x = int(xmin * w_img)
            y = int(ymin * h_img)

            w = int((xmax - xmin) * w_img)
            h = int((ymax - ymin) * h_img)

            area = w * h

            bboxs.append(
                [i,(x,y,w,h),scores[0][i]]
            )

            if area > largest_area:

                largest_area = area

                bbox = (x,y,w,h)

                center_x = x + w // 2

        if largest_area > 0:

            frame = self.fancyDraw(
                frame,
                bbox
            )

        return frame,bboxs,bbox,center_x


    def fancyDraw(self, frame, bbox):

        x,y,w,h = bbox

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        return frame


def open_camera():

    for idx in [0,1,2,3]:

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                print(f"Camera found: /dev/video{idx}")
                return cap

            cap.release()

    return None

def myTrack_Face_Follow():

    image = open_camera()

    if image is None:
        print("No camera found")
        return

    ret, frame = image.read()
    print("CAM TEST =", ret)

    person_detector = PersonDetector()

    ret, frame = image.read()
    print("AFTER SSD =", ret)

    if not image.isOpened():
        print("ERROR: Cannot open camera index 0")
        return

    image_width = 640
    image_height = 480

    image.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    image.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    imshow_num = 0
    
    pan_angle = float(PAN_CENTER)
    tilt_angle = float(TILT_CENTER)
    
    Facebot.Ctrl_Servo(1, int(pan_angle))
    Facebot.Ctrl_Servo(2, int(tilt_angle))
    
    try:
        while 1:
            ret, frame = image.read()
            print("READ =", ret)
            if not ret or frame is None:
                print("ERROR: Failed to read camera frame")
                stop_robot()
                reset_tracking_pid()
                break

            frame, bboxs, bbox, center_x = person_detector.findPersons(frame)
            x, y, w, h = bbox

            fall_state = fall_detect(w, h)
            if bboxs:
                if fall_state:

                    cv2.putText(
                        frame,
                        "FALL DETECTED",
                        (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3
                    )

                    print("******** FALL DETECTED ********")
                    
                print("PERSON DETECTED")
                now_aera = w*h
                now_aera = limit_max_vlaue(now_aera,2000,350000)#限制下面积的最小最大
                face_center_y = y + h // 2
                TARGET_X = image_width // 2
                pan_error = center_x - TARGET_X
                tilt_error = face_center_y - image_height // 2

                # Move the camera in small bounded steps. This is smoother than
                # converting each frame's PID output directly into an angle.
                if abs(pan_error) > PAN_DEAD_ZONE:
                    pan_step = clamp(
                        pan_error * PAN_GAIN,
                        -PAN_MAX_STEP,
                        PAN_MAX_STEP,
                    )
                    pan_angle = clamp(
                        pan_angle - pan_step,
                        PAN_MIN,
                        PAN_MAX,
                    )
                    Facebot.Ctrl_Servo(1, int(round(pan_angle)))

                if abs(tilt_error) > TILT_DEAD_ZONE:
                    tilt_step = clamp(
                        tilt_error * TILT_GAIN,
                        -TILT_MAX_STEP,
                        TILT_MAX_STEP,
                    )
                    tilt_angle = clamp(
                        tilt_angle - tilt_step,
                        TILT_MIN,
                        TILT_MAX,
                    )
                    Facebot.Ctrl_Servo(2, int(round(tilt_angle)))

                #电机X轴pid
                # Let the camera pan first. Rotate the chassis only when the
                # pan servo approaches an edge and cannot comfortably follow.
                pan_near_left_edge = pan_angle <= PAN_MIN + PAN_CHASSIS_MARGIN
                pan_near_right_edge = pan_angle >= PAN_MAX - PAN_CHASSIS_MARGIN
                chassis_turn_needed = (
                    (pan_error > PAN_DEAD_ZONE and pan_near_left_edge)
                    or (pan_error < -PAN_DEAD_ZONE and pan_near_right_edge)
                )

                if chassis_turn_needed:
                    direction_pid.SystemOutput = center_x
                    direction_pid.SetStepSignal(int(image_width/2))
                    direction_pid.SetInertiaTime(0.01, 0.1)
                    target_valuex = int(direction_pid.SystemOutput)
                else:
                    target_valuex = 0
                    reset_pid(direction_pid)

                raw_area = w*h
                print(
                    f"x={center_x} "
                    f"target_x={target_valuex} "
                    f"pan={pan_angle:.1f} "
                    f"tilt={tilt_angle:.1f} "
                    f"w={w} "
                    f"h={h} "
                    f"raw={raw_area} "
                    f"area={now_aera}"
                )

                #print(target_valuex)
                if target_valuex > -10 and target_valuex < 10:
                    target_valuex = 0 #剔除死区
                
                
                #根据面积进行前进后退
                speed_pid.SystemOutput = now_aera
                speed_pid.SetStepSignal(area_center)
                speed_pid.SetInertiaTime(0.01, 0.1)               
                speed_value = int(speed_pid.SystemOutput)
                #print(speed_value)
                if speed_value > -20 and speed_value < 20:
                    speed_value = 0 #增加静止区
                else:
                    #剔除死区
                    if speed_value<0: 
                        speed_value = limit_max_vlaue(speed_value,-12,-6)
                    else:
                        speed_value = limit_max_vlaue(speed_value,6,12)

                # Draw the desired face center and current face center.
                cv2.circle(
                    frame,
                    (image_width // 2, image_height // 2),
                    10,
                    (0, 0, 255),
                    -1,
                )

                cv2.line(
                    frame,
                    (image_width // 2, image_height // 2),
                    (center_x, face_center_y),
                    (0, 255, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    f"center={center_x}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"face=({center_x},{face_center_y})",
                    (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )
                
                
                #control_motor_speed(speed_value,-target_valuex)
                            
            
            else:
                stop_robot()
                reset_tracking_pid()
                
            imshow_num +=1
            if imshow_num%2==0:
                cv2.imshow("face", frame)
                imshow_num = 0
                
            if cv2.waitKey(1)==ord('q'):
                break
    except Exception as e:
        print("ERROR:", e)
        
        import traceback
        traceback.print_exc()
    finally:
        stop_robot()
        Facebot.Ctrl_Servo(1,90)
        Facebot.Ctrl_Servo(2,25)
        reset_tracking_pid()
        if image is not None:
            image.release()
        cv2.destroyAllWindows()
       
       
if __name__ == "__main__":
    myTrack_Face_Follow()
