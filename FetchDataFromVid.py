import cv2
import numpy as np
import time
from datetime import datetime, timedelta
import os
from centroidtracker import CentroidTracker
import copy
import json
import csv
import argparse

DATADICT = {}
FRAME = np.array([])

class FetchData:
    def __init__(self, fullPath, enableImshow=0, enableGPU=0, printData=0, printProgress=1, exportJSON=0, exportCSV=1):
        self.printData = printData
        self.fullPath = fullPath
        self.enableImshow = enableImshow
        self.exportJSON = exportJSON
        self.exportCSV = exportCSV
        self.printProgress = printProgress

        self.basepath, self.filename, self.videoStartDatetime = self.getStartDatetime(self.fullPath)
        self.tracker = CentroidTracker(maxDisappeared=20, maxDistance=90)
        self.net, self.faceNet, self.ageNet, self.genderNet = self.modelsInit(enableGPU)
        self.timestamp = self.videoStartDatetime

        self.DATADICT = {
            "timestamp": "",
            "screenTime": 0,
            "dwellTime": 0,
            "totalCount": 0,
            "impressions": 0,
            "males": 0,
            "females": 0,
            "young": 0,
            "middleAged": 0,
            "elderly": 0
        }

        if (exportJSON):
            data = {self.filename:[]}
            self.jsonPathName = os.path.join(self.basepath, self.filename+".json")
            self.write_json(data)

        if (exportCSV):
            data = ['timestamp','screenTime','dwellTime','totalCount','impressions','males','females','young','middleAged','elderly']
            self.csvPathName = os.path.join(self.basepath,self.filename+".csv")
            self.write_csv(data, 'w')

        self.run()

        if (exportJSON):
            print("JSON- ", self.jsonPathName)

        if (exportCSV):
            print("CSV- ", self.csvPathName)

    def write_csv(self, data, m):
        with open(self.csvPathName, mode=m) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(data)

    def write_json(self, data): 
        with open(self.jsonPathName,'w') as f: 
            json.dump(data, f, indent=2)

    def read_json(self):
        with open(self.jsonPathName) as f:
            return json.load(f) 

    def run(self):

        cap = cv2.VideoCapture(self.fullPath)
        video_framerate = cap.get(cv2.CAP_PROP_FPS)
        video_frametime = timedelta(seconds=(1/video_framerate))
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        object_id_list = []
        impressionsObjectId = []
        genderDecidedConfidence = {}
        ageDecidedConfidence = {}
        gender = {}
        age = {}
        total_count = 0
        screenTime_Time = dict()
        dwellTime_Time = dict()
        AccumulatedScreenTime = 0
        AccumulatedDwellTime = 0
        fps = 0
        frameNo = 0

        while True:
            start_time = time.time()

            ret, image = cap.read()
            if not ret:
                break

            image_cpy = copy.deepcopy(image)

            unique_count = 0
            current_count = 0

            h, w = image.shape[:2]
            
            DwellYLevel = h//2
            cv2.line(image, (0, DwellYLevel), (w, DwellYLevel), (255, 128, 0), 2)
            cv2.putText(image, "Dwell Area", (w - 250, DwellYLevel + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 128, 0), 1)

            blob = cv2.dnn.blobFromImage(image_cpy, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)

            layer_outputs = self.net.forward(ln)

            outputs = np.vstack(layer_outputs)

            boxes, confidences, class_ids, rects = [], [], [], []

            for output in outputs:

                scores = output[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == 0 and confidence > 0.5:

                    box = output[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)

            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    person_box = [x, y, x + w, y + h]
                    rects.append(person_box)

            boundingboxes = np.array(rects)
            boundingboxes = boundingboxes.astype(int)
            objects, dereg = self.tracker.update(rects)

            for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = abs(int(x1))
                y1 = abs(int(y1))
                x2 = abs(int(x2))
                y2 = abs(int(y2))

                centroidX = int((x1 + x2) / 2)
                centroidY = int((y1 + y2) / 2)

                cv2.circle(image, (centroidX, centroidY), 5, (255, 255, 255), thickness=-1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if objectId not in object_id_list:  # NEW PERSON
                    object_id_list.append(objectId)  # Append in ObjectID list - list of ID of people who have come before

                    unique_count += 1

                    screenTime_Time[objectId] = 0  # Initialize

                    dwellTime_Time[objectId] = 0 

                    genderDecidedConfidence[objectId] = 0
                    gender[objectId] = ""
                    ageDecidedConfidence[objectId] = 0
                    age[objectId] = ""

                else:  # EXISTING PERSON

                    screenTime_Time[objectId] += 1 / video_framerate
                    AccumulatedScreenTime += 1 / video_framerate
                
                    if centroidY > DwellYLevel:
                        dwellTime_Time[objectId] += 1 / video_framerate
                        AccumulatedDwellTime += 1 / video_framerate
                        if (objectId not in impressionsObjectId):
                            impressionsObjectId.append(objectId)
                            self.DATADICT["impressions"]+=1
                    
                                                                     
                    personCrop = image_cpy[y1:y2,x1:x2]
                    frapersonCropEnlarge = cv2.resize(personCrop, (0, 0), fx=2, fy=2)
                    # cv2.imshow("person",frapersonCropEnlarge)
                    ageGenderResult = self.detect_and_predict_age(frapersonCropEnlarge)
                    if (ageGenderResult):
                        if(ageGenderResult[0]["gender"][1]>genderDecidedConfidence[objectId]):
                            gender[objectId] = ageGenderResult[0]["gender"][0]
                            genderDecidedConfidence[objectId] = ageGenderResult[0]["gender"][1]
                        if(ageGenderResult[0]["age"][1]>ageDecidedConfidence[objectId]):
                            age[objectId] = ageGenderResult[0]["age"][0]
                            ageDecidedConfidence[objectId] = ageGenderResult[0]["age"][1]

            
                ##################################################

                # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                ID_text = "Person:" + str(objectId)
                ScreenTimeText = "DwellTime: {:.2f}".format(screenTime_Time[objectId]) + str("s")  # INTERCHANGED
                DwellTimeText = "ScreenTime: {:.2f}".format(dwellTime_Time[objectId]) + str("s")  # INTERCHANGED
                cv2.putText(
                    image,
                    ID_text,
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    image,
                    ScreenTimeText,
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    image,
                    DwellTimeText,
                    (x1, y1 - 60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (0, 0, 255),
                    2,
                )
                current_count += 1
            
            for l in list(dereg.items()): 
                # print(l)
                objectID = l[0]
                if (age[objectID] and gender[objectID]):
                    if (age[objectID] == "Young"):
                        self.DATADICT["young"]+=1
                    elif (age[objectID] == "Middle"):
                        self.DATADICT["middleAged"]+=1
                    elif (age[objectID] == "Elderly"):
                        self.DATADICT["elderly"]+=1
                    
                    if (gender[objectId] == "MALE"):
                        self.DATADICT["males"]+=1
                    elif (gender[objectId] == "FEMALE"):
                        self.DATADICT["females"]+=1
                        
                self.tracker.deleteDereg(l[0])

            total_count += unique_count
            self.DATADICT["totalCount"]=total_count
            total_count_text = "Total Count:" + str(total_count)
            cv2.putText(
                image,
                total_count_text,
                (5, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )

            current_count_text = "Current Count:" + str(current_count)
            cv2.putText(
                image,
                current_count_text,
                (5, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )

            self.DATADICT["screenTime"] = round(AccumulatedScreenTime / 60, 2)
            AccumulatedScreenTime_Text = "Total DwellTime: {:.2f}".format(AccumulatedScreenTime / 60) + str("m")
            cv2.putText(
                image,
                AccumulatedScreenTime_Text,
                (5, 550),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )  # INTERCHANGED

            self.DATADICT["dwellTime"] = round(AccumulatedDwellTime / 60,2)
            AccumulatedDwellTime_Text = "Total ScreenTime: {:.2f}".format(AccumulatedDwellTime / 60) + str("m")
            cv2.putText(
                image,
                AccumulatedDwellTime_Text,
                (5, 600),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )  # INTERCHANGED


            end_time = time.time()
            frameTime = (end_time - start_time)
            frameTimeDatetime = timedelta(seconds=frameTime)
            self.timestamp+=video_frametime
            self.DATADICT["timestamp"]=self.timestamp.strftime("%m/%d/%Y, %H:%M:%S")
            fps = 1 / frameTime
            fps_text = "FPS: {:.2f}".format(fps)
            cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frameNo += 1

            if (self.enableImshow):
                cv2.imshow("Application", image)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            
            ######################  GLOBAL VARIABLES    ######################
            global DATADICT, FRAME
            DATADICT = self.DATADICT.copy()
            FRAME = copy.deepcopy(image)

            if(self.printData):
                print(DATADICT)

            if(self.printProgress):
                print("Processed", frameNo, "of", totalFrames, "Frames |", round((frameNo/totalFrames)*100,2), "% Completion |", fps_text)

            if(self.exportJSON):
                data = self.read_json()
                data[self.filename].append(DATADICT)
                self.write_json(data)

            if(self.exportCSV):
                data = []
                for key in DATADICT:
                    data.append(DATADICT[key])
                self.write_csv(data, 'a')            

        cap.release()
        cv2.destroyAllWindows()
        
        return
    
    def detect_and_predict_age(self, frame):
        faceNet = self.faceNet
        ageNet = self.ageNet
        genderNet = self.genderNet
        minConf=0.5
        
        AGE_BUCKETS = [
            "(0-2)",
            "(4-6)",
            "(8-12)",
            "(15-20)",
            "(25-32)",
            "(38-43)",
            "(48-53)",
            "(60-100)",
        ]

        AGE_CATEGORY = {
            "Young":["(0-2)","(4-6)","(8-12)","(15-20)"],
            "Middle":["(25-32)","(38-43)"],
            "Elderly":["(48-53)","(60-100)"]
        }

        GENDER_BUCKETS = ["MALE", "FEMALE"]

        results = []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > minConf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                # ensure the face ROI is sufficiently large
                if face.shape[0] < 20 or face.shape[1] < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(
                    face,
                    1.0,
                    (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False,
                )

                ageNet.setInput(faceBlob)
                preds = ageNet.forward()
                i = preds[0].argmax()
                age = AGE_BUCKETS[i]
                
                for key in AGE_CATEGORY:
                    if (age in AGE_CATEGORY[key]):
                        ageCategory = key

                ageConfidence = preds[0][i]

                genderNet.setInput(faceBlob)
                genderPred = genderNet.forward()
                j = genderPred[0].argmax()
                gender = GENDER_BUCKETS[j]
                genderConfidence = genderPred[0][j]

                d = {
                    "loc": (startX, startY, endX, endY),
                    "age": (ageCategory, ageConfidence),
                    "gender": (gender, genderConfidence),
                }
                results.append(d)

        return results

    def getStartDatetime(self, path):
        directory = os.path.split(path)[0]
        basename = os.path.basename(path)
        filename = os.path.splitext(basename)[0]
        try:
            year = int(filename[1:5])
            month = int(filename[5:7])
            day = int(filename[7:9])
            hours = int(filename[9:11])
            minutes = int(filename[11:13])
            seconds = int(filename[13:15])
            videoStartDatetime = datetime(year, month, day, hours, minutes, seconds, 0)
        except:
            videoStartDatetime = datetime.now()
            print("Invalid Video Name for DateTime extraction. Setting start time to ", videoStartDatetime)
        return directory, filename, videoStartDatetime
    
    def modelsInit(self, enableGPU):
        labels = open("ModelsAndWeights/data/coco.names").read().strip().split("\n")
        net = cv2.dnn.readNetFromDarknet("ModelsAndWeights/cfg/yolov4.cfg", "ModelsAndWeights/yolov4.weights")

        faceNet = cv2.dnn.readNet(
            "ModelsAndWeights/face_detector/deploy.prototxt",
            "ModelsAndWeights/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        )
        
        ageNet = cv2.dnn.readNet(
            "./ModelsAndWeights/age_detector/age_deploy.prototxt",
            "./ModelsAndWeights/age_detector/age_net.caffemodel",
        )
        
        genderNet = cv2.dnn.readNet(
            "./ModelsAndWeights/gender_detector/gender_deploy.prototxt",
            "./ModelsAndWeights/gender_detector/gender_net.caffemodel",
        )
        
        if (enableGPU):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        return net, faceNet, ageNet, genderNet

    
def main():
    print("\n","\t",args,"\n")
    FetchData(fullPath=args["path"], enableImshow=args["imshow"], enableGPU=args["enableGPU"], printData=args["printData"], printProgress=args["printProgress"], exportJSON=args["exportJSON"], exportCSV=args["exportCSV"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Process videos for demographic data extraction.')
    ap.add_argument("-path", "--path", required=True, help="Full path to video")
    ap.add_argument("-show", "--imshow", required=False, type = int, default = 0, help="Display processed video frames")
    ap.add_argument("-gpu", "--enableGPU", required=False, type = int, default = 0, help="Enable or Disable GPU based processing")
    ap.add_argument("-data", "--printData", required=False, type = int, default = 0, help="Print extracted data")
    ap.add_argument("-prog", "--printProgress", required=False, type = int, default = 1, help="Display progress")
    ap.add_argument("-json", "--exportJSON", required=False, type = int, default = 0, help="Export data in JSON file")
    ap.add_argument("-csv", "--exportCSV", required=False, type = int, default = 1, help="Export data in CSV file")
    args = vars(ap.parse_args())
    main()