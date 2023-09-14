import supervision as sv
import cv2

generator = sv.get_video_frames_generator("./mall.mp4")
iterator = iter(generator)
frame = next(iterator)

cv2.imwrite("frame.jpg", frame)
