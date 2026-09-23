from ultralytics import YOLO

model_obj = YOLO("./models/yolov8n.pt")
model_obj.export(format="onnx")  

model_pose = YOLO("./models/yolov8n-pose.pt")
model_pose.export(format="onnx") 
