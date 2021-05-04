# Demographic Data from Video | YOLOv4, AlexNet, ResNet10-SSD

## Table of Contents

- [About](#about)
- [Data Description](#data)
- [Prerequisites](#prereq)
- [Folder Structure](#folder)
- [Running the script](#install)
- [Modules used](#dl)
- [Contributors](#contrib)

## About <a name = "about"></a>

This project is used to extract demographic data from CCTV footage videos for the purpose of advertisements.

## Data Description <a name = "data"></a>

The following data can be extracted from the video-
- Count of people
- Impressions
- Time spent
  - Screen Time : Total time spent in front of the camera.
  - Dwell Time : Time spend under the lower half of the camera. (Assuming attention)
- Gender
  - Male 
  - Female
- Age
  - Young (0-20)
  - Middle Aged (20-50)
  - Elderly (50 and above)

## Prerequisites <a name = "prereq"></a>

A good GPU is preferrable to process the videos.
Hence CUDA setup is required along with GPU build of openCV.

```
opencv==4.5.1
numpy==1.19.1
```

## Folder Structure and Files Required <a name = "folder"></a>

```
├── centroidtracker.py
├── FetchDataFromVid.py
├── ModelsAndWeights
│   ├── age_detector
│   │   ├── age_deploy.prototxt
│   │   └── age_net.caffemodel
│   ├── cfg
│   │   ├── yolov4.cfg
│   │   └── yolov4-tiny.cfg
│   ├── data
│   │   └── coco.names
│   ├── face_detector
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   ├── gender_detector
│   │   ├── gender_deploy.prototxt
│   │   └── gender_net.caffemodel
│   └── yolov4.weights
├── README.md
├── requirements.txt
└── videos
    ├── L20201015151214400.mp4
    ├── L20201210134503431.csv
    └── L20201210134503431.mp4
```

Place the videos inside videos folder.
Download .caffemodel and yolov4.weights files from the internet.
## Running the script<a name = "install"></a>

save the videos in videos/ folder. Run the following-

```
python FetchDataFromVid.py -path "./videos/L20201210134503431.mp4"
```
Arguments for FetchDataFromVid.py
```
-path   # Full path to video
-show   # Display processed video frames
-gpu    # Enable or Disable GPU based processing
-data   # Print extracted data
-prog   # Display progress
-json   # Export data in JSON file
-csv    # Export data in CSV file
```

## Modules used <a name = "dl"></a>

#### YOLOv4
For person detection

#### ResNet10 SSD
For face detection

#### AlexNet
For age and gender prediction

#### Centroid Tracker
For object tracking


## Contributors <a name = "contrib"></a>

#### <a href="https://www.linkedin.com/in/sakshamprakash/">Saksham Prakash</a>
