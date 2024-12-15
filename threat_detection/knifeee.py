from ultralytics import YOLO
import cv2
import os
import pygame


pygame.mixer.init()


model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Eh oh - Camera not found")
    exit()

THREAT_CLASSES = ['knife', 'gun', 'weapon', 'suspicious object']


def alert_sound():
    pygame.mixer.music.load("alert.wav")
    pygame.mixer.music.play()

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        print("Wot ma - Frame not captured")
        break

    results = model(frame)


    annotated_frame = results[0].plot()

 
    detected_threats = [results[0].names[int(cls)] for cls in results[0].boxes.cls if results[0].names[int(cls)] in THREAT_CLASSES]
    
    if detected_threats:
        alert_message = f"Threats detected: {', '.join(detected_threats)}"
        print(alert_message)
        alert_sound()

   
        with open("detected_threats.txt", "a") as file:
            file.write(alert_message + "\n")


    cv2.imshow("YOoooo - Threat Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
