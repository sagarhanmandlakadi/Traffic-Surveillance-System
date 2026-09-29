import cv2
import torch
import os
import easyocr

video_filename = "traffic.mp4.mp4"
if not os.path.exists(video_filename):
    print(f"❌ ERROR: Cannot find '{video_filename}'")
    exit()

print("🚀 Initializing Dynamic Tracking Traffic Surveillance System...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
reader = easyocr.Reader(['en'], gpu=False) 

cap = cv2.VideoCapture(video_filename)
vehicle_count = 0
already_counted = set()

frame_counter = 0
ocr_interval = 12  # Increased interval slightly to completely eliminate lagging
cached_plate_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    height, width, _ = frame.shape
    line_y = int(height * 0.6)

    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 3)
    cv2.putText(frame, f"Total Count: {vehicle_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if frame_counter % 2 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]

        for index, row in detections.iterrows():
            if row['confidence'] > 0.25:
                original_label = row['name']
                display_label = None

                if original_label in ['car', 'truck', 'van']:
                    display_label = "Vehicle"
                elif original_label in ['bus']:
                    display_label = "Heavy Vehicle"
                elif original_label in ['motorcycle', 'bicycle']:
                    display_label = "Two-Wheeler"

                if display_label is not None:
                    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)

                    # --- ADAPTIVE BUMPER CROPPING FIX ---
                    if abs(cy - line_y) < 60 and (ymax > ymin):
                        # Adjust crop height dynamically based on vehicle type
                        if display_label == "Two-Wheeler":
                            # Motorcycles have plates lower down near the rear tire
                            plate_ymin = ymin + int((ymax - ymin) * 0.65)
                        elif original_label == "truck":
                            # Large commercial trucks often have front plates mid-grill
                            plate_ymin = ymin + int((ymax - ymin) * 0.40)
                        else:
                            # Standard sedans, SUVs, and hatchbacks
                            plate_ymin = ymin + int((ymax - ymin) * 0.55)

                        crop_ymin, crop_ymax = max(0, plate_ymin), min(height, ymax)
                        crop_xmin, crop_xmax = max(0, xmin), min(width, xmax)

                        if (crop_ymax > crop_ymin) and (crop_xmax > crop_xmin):
                            plate_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                            resized_gray = cv2.resize(plate_crop, (300, 100))
                            
                            cv2.imshow("Target License Plate Region", resized_gray)

                            # Throttled execution prevents computing empty image arrays constant loops
                            if frame_counter % ocr_interval == 0:
                                gray_scan = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                ocr_result = reader.readtext(gray_scan)
                                if ocr_result:
                                    # Extract string elements safely
                                    words = [res[1] for res in ocr_result]
                                    cached_plate_text = " ".join(words).strip()

                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    label_str = f"{display_label} [{cached_plate_text}]" if cached_plate_text else display_label
                    cv2.putText(frame, label_str, (xmin, ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if abs(cy - line_y) < 6:
                        pos_id = f"{cx}-{cy}"
                        if pos_id not in already_counted:
                            vehicle_count += 1
                            already_counted.add(pos_id)
                            cached_plate_text = ""

    cv2.imshow("Traffic Surveillance & Counting System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()