from roboflow import Roboflow
rf = Roboflow(api_key="UroOOULqNYBZQAehTDQv")
project = rf.workspace("kzl").project("ronaldo-x-messi-detection")
version = project.version(1)
dataset = version.download("yolov8")