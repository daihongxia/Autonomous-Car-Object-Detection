# Autonomous Car Object 

This project is a implementation of CenterNet algorithm in the object detection and pose prediction set up.
## Problem
This project uses a dataset published by Baidu and PKU that includes thousands of photos taken on top of a self-driving car on the street. The task is to identify other vehicles in the photos and also extract their 3D coordinates and movement (Yaw-Roll-Pitch). 

## Performance
It's hard to quantify the performance for this type of tasks. Here are some example photos comparing the prediction from the CenterNet model on a validation dataset, and the ground truth. This shows that the model is generally performing pretty well. It can even identify certain vehicles in the photos that can barely be recognized by human eyes.
![comparison](example.png)