import cv2

file_path = './exam_video/test1.mp4'
cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        exit()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    count += 1
    cv2.imwrite(f'./test_{count}.jpg', frame)
