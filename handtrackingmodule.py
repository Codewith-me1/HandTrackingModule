


import cv2
import mediapipe as mp 
import time
import mediapipe.python.solutions.hands
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import mediapipe.python.solutions.drawing_utils
import mediapipe.python.solutions.hands
import mediapipe.python.solutions.hands_connections
import mediapipe.python.solutions.holistic
import mediapipe.python.solutions.objectron
import mediapipe.python.solutions.pose
import mediapipe.python.solutions.selfie_segmentation
import mediapipe.python.solutions.drawing_utils
import mediapipe.python.solutions.face_detection
import mediapipe.python.solutions.face_mesh
import mediapipe.python.solutions.face_mesh_connections
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from typing import NamedTuple


class hand():
     def __init__(self,static_image_mode =False, max_num_hands= 2, model_complexity =1, min_detection_confidence=0.5,  min_tracking_confidence= 0.5) :
        self.mode = static_image_mode
        self.maxHands =  max_num_hands
        self.detectionCon =  min_detection_confidence
        self.trackCon = min_tracking_confidence
        self.modelcom =  model_complexity

        self.mpHands = mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelcom ,self.detectionCon,self.trackCon)
        self.mpDraw =  mediapipe.python.solutions.drawing_utils
        
     def findHands(self,frame, draw = True):
         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         self.results = self.hands.process(imgRGB)
         
        # print(results.multi_hand_world_landmarks)

         if self.results.multi_hand_landmarks:
             for handLns in self.results.multi_hand_landmarks:
                 if draw:
                   self.mpDraw.draw_landmarks(frame, handLns, self.mpHands.HAND_CONNECTIONS)
   
         return frame

     def findPosition(self,frame, handNo=0, draw = True):    

       lmList = []  
       if  self.results.multi_hand_landmarks:
         myhand=   self.results.multi_hand_landmarks[handNo]

         for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                # print( id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                   cv2.circle(frame, (cx,cy), 5, (255,0,255),cv2.FILLED)

       return lmList                

def main():

            pTime = 0
            cTime = 0
            cap = cv2.VideoCapture(0)
            detector = hand()
            while(True):
      
                ret, frame = cap.read()
                frame= detector.findHands(frame)
                lmList= detector.findPosition(frame)
                if len(lmList) != 0:
                  print(lmList[4],[8])

                cTime = time.time()
                fps =1 / (cTime - pTime)
                pTime = cTime

                cv2.putText(frame,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,( 255,0,255),3)
                                                                        

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
            frame.release
            cv2.destroyAllWindows()

if __name__=="__main__":
           main()
 
    