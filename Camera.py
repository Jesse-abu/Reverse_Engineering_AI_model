import cv2
import pandas as pd
import json
from ultralytics import YOLO

file = pd.read_csv(r'csv_files/everyday_electronics/personal_gadgets.csv')

model = YOLO("runs\\detect\\train14\\weights\\best.pt")

cam = cv2.VideoCapture(0)
text_data = ''
entries = 0

with open('item_data', 'r') as f:
    data = json.load(f)
    f.close()

print(data["around_the_house"])

if not cam.isOpened():
    print('Failed to open camera')
    exit()


print('Exit button: q, reset_output: r')

def put_multiline_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=1.0, color=(255,255,255), thickness=2, line_spacing=4):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        y = org[1] + i * (h + baseline + line_spacing) + h
        cv2.putText(img, line, (org[0], y), font, font_scale, color, thickness, cv2.LINE_AA)

def results(img, color, text):
    results = model(img[..., ::-1])


    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f'{model.names[class_id]}: {confidence:.2f}'
            lbl = ''.join(model.names[class_id].split()).capitalize()
            if lbl in data['everyday_electronics']['personal']:
                for row in range(len(file)):
                    if file.loc[row]['Gadget'] == lbl:
                        text += f'{file.loc[row]['Component']}: {file.loc[row]['Detailed_Function']}\n'
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        put_multiline_text(img, text, (10, 10), font_scale=0.5, color=color)

    
        

cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('detector', 1280, 720)

while True:
    retrieve, frame = cam.read()
    color = (0, 255, 0)
    
    results(frame, color, text_data)

    if not retrieve:
        print('failed to grab frame')
        break

    cv2.imshow('detector', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        text_data = ''    
        pass

cam.release()
cv2.destroyAllWindows()