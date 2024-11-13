import cv2
import random
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

CONFIDENCE_THRESHOLD = 0.6  

def draw_text_with_shadow(image, text, position, font_scale, color, thickness, shadow_color, offset):
    cv2.putText(image, text, (position[0] + offset, position[1] + offset), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness + 2)
    cv2.putText(image, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]
        confidence = r.conf[0]
        class_id = int(r.cls[0])
        label = model.names[class_id]
        if confidence >= CONFIDENCE_THRESHOLD:
            detections.append((x1, y1, x2, y2, confidence, label))

    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    detections = detections[:4]

    for x1, y1, x2, y2, confidence, label in detections:
        translated_label = label  # No estamos traduciendo en este caso, pero puedes agregarlo si lo deseas
        color = random_color()  # Genera un color aleatorio
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    
        draw_text_with_shadow(frame, f"{translated_label} {confidence:.2f}", 
                              (int(x1), int(y1) - 10), 0.7, (255, 255, 255), 2, (0, 0, 0), 2)
    cv2.imshow("Detección de Objetos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()