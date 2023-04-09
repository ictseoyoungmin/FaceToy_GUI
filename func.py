import time
import math
import numpy as np
import cv2

import mediapipe as mp

# Load the face mesh model
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10,min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)

# bounding box test
def test_m1(frame):
    re = faceMesh.process(frame)
    frame = np.zeros_like(frame)
    if re.multi_face_landmarks:
        points = []
        for landmark in re.multi_face_landmarks[0].landmark: # normalized x,y,z points
            points.append(normalized_to_pixel_coordinates(landmark.x,landmark.y,frame.shape[1],frame.shape[0]))
        if None in points:
            return frame
        points = np.array(points,dtype=np.int16)
        for i,l in enumerate(points):    
            cv2.putText(frame,f'{i}',l,cv2.FONT_HERSHEY_PLAIN,0.6, (0,255,0),1)
    
    return frame

# lib test
def test_m2(frame):
    frame = swapFace(frame)
    return frame

def test_m3(frame):
    re = faceMesh.process(frame)
    frame = np.zeros_like(frame)
    if re.multi_face_landmarks:
        for faceLms in re.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                    drawSpec,drawSpec)
        
    return frame


def processing(frame,handler):
    methods = [test_m1,test_m2,test_m3]
    return methods[handler](frame)

def plot_fps(frame,pTime,loc=(20,80),font_face=cv2.FONT_HERSHEY_PLAIN,font_scale=1,color=(255,0,0),thinkness=2):
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f'FPS: {int(fps)}',loc,font_face,font_scale,color,thinkness)
    return frame,pTime

# mpDraw.draw_landmarks() 함수 일부 문장 복사
def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) :
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# face landmarks detection and get points
def getPoints(img,cvt_color=None,return_cvt_img=False):
    img_rows,img_cols = img.shape[:2]
    if cvt_color:
        img = cv2.cvtColor(img,cvt_color)
         
    result = faceMesh.process(img) # one landmarks of faces 
    
    # landmarks x,y pair
    points = []
    if result.multi_face_landmarks:
        for landmark in result.multi_face_landmarks[0].landmark: # normalized x,y,z points
            points.append(normalized_to_pixel_coordinates(landmark.x,landmark.y,img_cols,img_rows))
            # print(landmark.x,landmark.y)
    if return_cvt_img:
        return points, img
    else:
        return points
    
# Delaunay triangulation
def getTriangles(img,points):
    h,w = img.shape[:2]
    subdiv = cv2.Subdiv2D((0,0,w,h));
    subdiv.insert(points)
    triangleList = subdiv.getTriangleList()
    triangles = []
    for t in triangleList:
        pt = t.reshape(-1,2)
        if not (pt < 0).sum() and not (pt[:,0] > h).sum() \
                              and not (pt[:,1] > w).sum():
            indice = []
            for i in range(0,3):
                for j in range(0,len(points)):
                    if abs(pt[i][0]-points[j][0]) < 1.0 \
                        and abs(pt[i][1]-points[j][1]) < 1.0:
                        indice.append(j)
            if len(indice) == 3:
                triangles.append(indice)
    
    return triangles

def warpTriangle(img1,img2,pts1,pts2):
    x1,y1,w1,h1 = cv2.boundingRect(np.float32([pts1])) # x,y : left-top points / w,h : length
    x2,y2,w2,h2 = cv2.boundingRect(np.float32([pts2]))

    roi1 = img1[y1:y1+h1, x1:x1+w1]
    roi2 = img2[y2:y2+h2, x2:x2+w2]
    
    offset1 = np.zeros((3,2),dtype=np.float32)
    offset2 = np.zeros((3,2),dtype=np.float32)
    for i in range(3):
        offset1[i][0], offset1[i][1] = pts1[i][0]-x1,pts1[i][1]-y1
        offset2[i][0], offset2[i][1] = pts2[i][0]-x2,pts2[i][1]-y2

    mtrx = cv2.getAffineTransform(offset1,offset2)
    
    warped = cv2.warpAffine(roi1,mtrx,(w2,h2),None,cv2.INTER_LINEAR,cv2.BORDER_REFLECT101)
    mask = np.zeros((h2,w2),dtype=np.uint8)
    cv2.fillConvexPoly(mask,np.int32(offset2),(255))
    
    warped_masked = cv2.bitwise_and(warped,warped,mask=mask)
    roi2_masked = cv2.bitwise_and(roi2,roi2,mask=cv2.bitwise_not(mask))
    roi2_masked = roi2_masked + warped_masked
    
    img2[y2:y2+h2, x2:x2+w2] = roi2_masked
    
def swapFace(frame,source_path='./dev/images.jpg'):
    source_img = cv2.imread(source_path)
    
    points_frame = getPoints(frame) 
    points_source,source_img = getPoints(source_img,cv2.COLOR_BGR2RGB,True)
    
    frame_draw = frame.copy()
    
    if np.shape(points_frame) != (468,2):
        return frame
    hullInedx = cv2.convexHull(np.array(points_frame),returnPoints=False)
    hull1 = [points_source[int(idx)] for idx in hullInedx]
    hull2 = [points_frame[int(idx)] for idx in hullInedx]
    
    triangles = getTriangles(frame,hull2)
    
    for i in range(0,len(triangles)):
        t1 = [hull1[triangles[i][j]] for j in range(3)]
        t2 = [hull2[triangles[i][j]] for j in range(3)]
        warpTriangle(source_img,frame_draw,t1,t2)
        
    mask = np.zeros(frame.shape[:2],dtype=frame.dtype)
    cv2.fillConvexPoly(mask,np.int32(hull2),(255,255,255))
    r = cv2.boundingRect(np.float32(hull2))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    output = cv2.seamlessClone(np.uint8(frame_draw),frame,mask,center,cv2.NORMAL_CLONE)
    
    return output