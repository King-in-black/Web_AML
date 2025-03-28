# Intelligent referee system 

## Overview
 The development of artificial intelligence (AI) has made it possible to address
 previously intractable challenges in modern sports competitions. In fencing, points
 are awarded when a player strikes their opponent with a sabre. However, determin
ing which fencer scores becomes difficult when both land touches simultaneously, as
 the decision relies on the rule of right of way. These events occur within milliseconds,
 making it nearly impossible for the human eye to capture the precise moment.
 To address this challenge, we designed and implemented an intelligent fencing
 system capable of accurately determining the right of way. The system collects
 acceleration data from four joints—the left and right elbows and ankles—using
 inertial measurement units (IMUs), and transmits this data via an ESP32-based
 wireless communication network.
 We constructed a dataset of 400 motion samples—covering attacks, forward
 movements, backward movements, and stops—collected from all team members to
 ensure diversity. The system consists of two main components: a front-end neural
 network based on Graph Attention Networks combined with 2 bidirectional LSTM
 layers for motion recognition, and a back-end state machine that determines the
 correct right of way based on official fencing rules. The front-end network classifies
 four distinct movements within 10 milliseconds, achieving an average accuracy of
 93.4%. Offline evaluations confirm the system’s ability to accurately assign the right
 of way, demonstrating its effectiveness.
 This novel approach to judging right of way improves fairness in fencing compe
titions while reducing the burden on human referees.


The final deliverables include:
---
- 📷 Illustrations of the design and results
- 💻 Code and design files
- 🎥 Video demonstration
- 📊 Dataset and visualizations

---
## Illustrations
Our team builds a wonderful wireless communication system based on ESP32 and IMUs .

![Demo Image](images/circuit_image.png)
![Another Image](images/img.png)

---
## Dataset Files

The sample data structure is like this:
We provide multiple CSV files for different motions and conditions. You can download them below:
- [any-backward.csv](data%2Fany-backward.csv)
- [any-forward.csv](data%2Fany-forward.csv)
- [100_attack.csv](data%2F100_attack.csv)
- [100_attack_processed.csv](data%2F100_attack_processed.csv)
- [100_backward.csv](data%2F100_backward.csv)
- [100_backward_processed.csv](data%2F100_backward_processed.csv)
- [100_forward.csv](data%2F100_forward.csv)
- [100_forward_processed.csv](data%2F100_forward_processed.csv)
- [100_stop.csv](data%2F100_stop.csv)[stop.csv](data%2Fstop.csv)
- [100_stop_processed.csv](data%2F100_stop_processed.csv)
---
And others are all inside the [data](data)
## Code and Design Files
### Hardware:

ESP32 sender:
- [sender_final.ino](code%2Fsender_final%2Fsender_final.ino)

ESP32 reciever:

- [receiver_final.ino](code%2Freceiver_final%2Freceiver_final.ino)

3D design box:
- [AMLLABBOX whole version final.SLDPRT](code%2FAMLLABBOX%20whole%20version%20final.SLDPRT)

3D design cover
- [box cover V3.0.SLDPRT](code%2Fbox%20cover%20V3.0.SLDPRT)
### Software:
train data recorder:

- [training_data_collection.py](code%2Ftraining_data_collection.py)

test data recorder:

- [test_dataset_collection.py](code%2Ftest_dataset_collection.py)

Python listener and data recorder:

- [python_listener.py](code%2Fpython_listener.py)

Preprocess:

- [preprocess.py](code%2Fpreprocess.py)

Load the preprocessed dataset:
- [Data_loader.py](code%2FData_loader.py)

Model inference code:
- [inference.py](code%2Finference.py)

Training code:

- [LSTM.py](code%2FLSTM.py)

Model weight:
- [bilstm_model.pth](code%2Fbilstm_model.pth)

Live prediction(for demo):

- [predict.py](code%2Fpredict.py)

---

## Slides and Video
Our final presentation:
- [📄 Presentation Slides](AFK_final_2025_03_21.pptx)

Our 2 minutes video:
- [▶️ Video Demo](AFK_video.mp4)

---
