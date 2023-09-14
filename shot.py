import supervision as sv
import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, cap_frame = cap.read()
    cv2.imshow("Shot", cap_frame)
    if (cv2.waitKey(30) == 27):
        cv2.imwrite("frame.jpg", cap_frame)
        break

cap.release()
cv2.destroyAllWindows()
