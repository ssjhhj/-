from ultralytics import YOLO

yolo = YOLO("./runs/detect/train20/weights/best.pt", task="detect")

# result = yolo(source="./ultralytics/assets/bus.jpg", conf=0.05)

result = yolo.predict(source='screen',conf=0.3) # 摄像头

# print(result[0].boxes)

# 将视频预测代码
# yolo detect predict model=runs/detect/train6/weights/best.pt source=./ceshi.mp4 show=True
# yolo detect predict model=best.pt source=./video/ceshi.mp4 show=True
# yolo detect predict model=best.pt source=./video/test9.mp4 show=True conf=0.5 augment="True" iou=0.6 vid_stride=3 hide_labels="True"
yolo detect predict model=BCD.pt source=./video/end.mp4 show=True conf=0.5 augment="True" iou=0.6 vid_stride=3

yolo detect predict model=BCD.pt source=./video/ceshi.mp4 show=True conf=0.5 augment="True" iou=0.2 line_width=3 hide_labels="False" boxes="False" show_conf="True" retina_masks="True" agnostic_nms="True" classes=[0,1]