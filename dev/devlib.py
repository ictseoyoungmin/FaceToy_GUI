# for f1.ipynb
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

__all__ = ['imgs_read_rgb',
           'imgs_get_landmarks',
           'normalized_to_pixel_coordinates',
           'rotate_img',
           'getPoints',
           'getTriangles',
           'warpTriangle',
           'get_connection_points',
           'vis_coordinates',
           'get_idx_to_coordinates',
           'get_eye_center',
           'get_angle',
           'align',
           'vis_triangle_list',
           'get_triangulation',
           'get_triangle_list',
           'morph_images',
           'get_masked_frame'
           ]

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10,min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)


def imgs_read_rgb(*img_pathes):
    imgs = []
    for img_path in img_pathes:
        imgs.append(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
    return imgs

def imgs_get_landmarks(*imgs):
    landmarks_list = []
    for img in imgs:
        landmarks_list.append(getPoints(img,None))
    return landmarks_list
     
def morph_images(image1, image2, landmarks1, landmarks2, alpha=0.5):
    """
    Given two input images and corresponding landmarks, morphs the images into each other based on the specified alpha
    value, which controls the degree of morphing.
    """
    if type(landmarks1) is not np.ndarray:
         landmarks1 = np.array(landmarks1)
         landmarks2 = np.array(landmarks2)
    # Find Delaunay triangulation for points in both images
    triangulation = get_triangulation(landmarks1, landmarks2)
    
    # Create output image as a copy of image1
    morphed_image = np.copy(image1)
    
    # Loop through each triangle in the triangulation
    for triangle in triangulation:
        # Get coordinates of the three vertices in the two images
        x1, y1 = landmarks1[triangle[0]], landmarks1[triangle[1]], landmarks1[triangle[2]]
        x2, y2 = landmarks2[triangle[0]], landmarks2[triangle[1]], landmarks2[triangle[2]]
        
        # Calculate the affine transformation matrix for each triangle
        affine_matrix = cv2.getAffineTransform(np.float32([(x1[0], y1[0]), (x1[1], y1[1]), (x1[2], y1[2])]),
                                                np.float32([(x2[0], y2[0]), (x2[1], y2[1]), (x2[2], y2[2])]))
        
        # Create a mask for the triangle
        mask = np.zeros(image1.shape[:2], dtype=np.float32)
        cv2.fillConvexPoly(mask, np.array([(x1[0], y1[0]), (x1[1], y1[1]), (x1[2], y1[2])]), 1)
        
        # Warp the triangle from image1 to image2 using the affine transformation and mask
        warped_triangle = cv2.warpAffine(image1, affine_matrix, (image1.shape[1], image1.shape[0]), None,
                                         cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        warped_triangle = warped_triangle * mask[:, :, np.newaxis]
        
        # Warp the triangle from image2 to image1 using the affine transformation and mask
        inverse_affine_matrix = cv2.getAffineTransform(np.float32([(x2[0], y2[0]), (x2[1], y2[1]), (x2[2], y2[2])]),
                                                        np.float32([(x1[0], y1[0]), (x1[1], y1[1]), (x1[2], y1[2])]))
        warped_triangle2 = cv2.warpAffine(image2, inverse_affine_matrix, (image2.shape[1], image2.shape[0]), None,
                                          cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        warped_triangle2 = warped_triangle2 * mask[:, :, np.newaxis]
        
        # Add the two warped triangles together and blend them based on alpha value
        blended_triangle = (1 - alpha) * warped_triangle + alpha * warped_triangle2
        
        # Replace the triangle in the morphed image with the blended triangle
        morphed_image_roi = morphed_image[min(y1):max(y1), min(x1):max(x1)]
        blended_triangle_roi = blended_triangle[min(y1):max(y1), min(x1):max(x1)]
        morphed_image_roi[np.where(blended_triangle_roi != 0)] = blended_triangle_roi[np.where(blended_triangle_roi != 0)]
        morphed_image_roi[min(y1):max(y1), min(x1):max(x1)][blended_triangle_roi != 0] = blended_triangle_roi[blended_triangle_roi != 0]

    return morphed_image_roi

def vis_triangle_list(triangle_list,w=178,h=119):
    # Create an empty image
    img = np.zeros((h, w, 3), np.uint8)

    # Draw each triangle in a random color
    for i in range(len(triangle_list)):
        # Get the indices of the three vertices of the triangle
        triangle = triangle_list[i]
        pt1 = (np.int32(triangle[0]), np.int32(triangle[1]))
        pt2 = (np.int32(triangle[2]), np.int32(triangle[3]))
        pt3 = (np.int32(triangle[4]), np.int32(triangle[5]))

        # Choose a random color for this triangle
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        # Fill the triangle with the color
        cv2.fillConvexPoly(img, np.array([pt1, pt2, pt3],dtype=np.int32), color)
        # Draw the boundary of the triangle
        # cv2.polylines(img, [pt1, pt2, pt3], , (0, 0, 0), thickness=1)
    plt.imshow(img)
    plt.show()
    return img

def get_triangle_list(landmarks1, landmarks2):
    # Create an empty rectangle that covers both images
    rectangle = (0, 0, max(landmarks1.max(axis=0)[0], landmarks2.max(axis=0)[0])+1,
                 max(landmarks1.max(axis=0)[1], landmarks2.max(axis=0)[1])+1)
    # Create a subdiv object
    subdiv = cv2.Subdiv2D(rectangle)
    # Insert landmarks into subdiv object
    for landmarks in [landmarks1, landmarks2]:
        for point in landmarks:
            subdiv.insert(np.int16((point[0], point[1])))

    # Get Delaunay triangulation as a list of triangles
    triangle_list = subdiv.getTriangleList()
    
    return triangle_list

def get_triangulation(landmarks1, landmarks2):
    """
    Given two sets of corresponding landmarks, computes and returns their Delaunay triangulation.
    """
    
    triangle_list = get_triangle_list(landmarks1, landmarks2)
    # Create a list of triangle indices based on the landmark indices
    delaunay_triangles = []
    for triangle in triangle_list:
        indices = []
        for i in range(3):
            for j, pt in enumerate([landmarks1, landmarks2]):
                if np.allclose(triangle[i*2], pt[:,0]) and np.allclose(triangle[i*2+1], pt[:,1]):
                    indices.append(j * len(landmarks1) + np.where((pt[:, 0] == triangle[i*2]) &
                                                                  (pt[:, 1] == triangle[i*2+1]))[0][0])
                    print('s')
        if len(indices) == 3:
            delaunay_triangles.append(tuple(indices))
            
    return delaunay_triangles     
     
        
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
    points = []
    if result.multi_face_landmarks:
        for landmark in result.multi_face_landmarks[0].landmark: # normalized x,y,z points
            points.append(normalized_to_pixel_coordinates(landmark.x,landmark.y,img_cols,img_rows))
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
    

### face align ##########################################################       

def vis_coordinates(img,idx_to_coordinates,connection=mpFaceMesh.FACEMESH_RIGHT_EYE):
    a = np.zeros_like(img)
    for conn in connection:
        start_idx = conn[0]
        end_idx = conn[1]
        cv2.line(a,idx_to_coordinates[start_idx],idx_to_coordinates[end_idx],color=(255,255,255))        
        # cv2.circle(a,idx_to_coordinates[start_idx],radius=1, color=(255,255,255))        
        # cv2.circle(a,idx_to_coordinates[end_idx],radius=1,color=(255,255,255))        

    plt.imshow(a)

def rotate_img(img,angle=15):
    height, width = img.shape[:2]
    center = (width/2, height/2)

    # 회전을 위한 변환 행렬 구하기
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 이미지 회전하기
    out_img = cv2.warpAffine(img, M, (width, height))
    # plt.imshow(out_img)
    return out_img

# FACEMESH_RIGHT_EYE(32,2,2)
def get_connection_points(idx_to_coordinates,connection):
    return np.array([[idx_to_coordinates[conn[0]],idx_to_coordinates[conn[1]]] for conn in connection]).reshape(-1,2)

def vis_coordinates(img,idx_to_coordinates,connection=mpFaceMesh.FACEMESH_LEFT_EYE):
    a = np.zeros_like(img)
    for conn in connection:
        start_idx = conn[0]
        end_idx = conn[1]
        cv2.line(a,idx_to_coordinates[start_idx],idx_to_coordinates[end_idx],color=(255,125,0))        
        # cv2.circle(a,idx_to_coordinates[start_idx],radius=1, color=(255,255,255))        
        # cv2.circle(a,idx_to_coordinates[end_idx],radius=1,color=(255,255,255))        
    plt.imshow(a)
    return a

def get_masked_frame(frame,idx_to_coordinates,connection=mpFaceMesh.FACEMESH_LEFT_EYE,color=(0,0,0),thickness=8):
    # mask = np.zeros(frame.shape[:2],dtype=frame.dtype)
    # cv2.fillConvexPoly(mask,np.int32(get_connection_points(idx_to_coordinates,connection)),(255,255,255))
    # frame = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.drawContours(frame,[get_connection_points(idx_to_coordinates,connection)],0,color,thickness)
    return frame

def get_idx_to_coordinates(img):
    '''
    a = np.zeros_like(img)   
     
    for conn in mpFaceMesh.FACEMESH_CONTOURS:
    start_idx = conn[0]
    end_idx = conn[1]
    cv2.line(a,idx_to_coordinates[start_idx],idx_to_coordinates[end_idx],color=(255,255,255))        
    \ncv2.circle(a,idx_to_coordinates[start_idx],radius=1, color=(255,255,255))        
    \ncv2.circle(a,idx_to_coordinates[end_idx],radius=2,color=(255,255,255))        

    plt.imshow(a)
    '''
    scale = 1
    image_cols, image_rows = img.shape[1]*scale,img.shape[0]*scale 
    landmark_list = []
    with mp.solutions.face_mesh.FaceMesh() as face_mesh:
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_list.append(face_landmarks)
        else:
            return None        
    idx_to_coordinates = {}
    points = []
    for idx,landmark in enumerate(landmark_list[0].landmark):
        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                    image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
            points.append(landmark_px)
    if idx_to_coordinates:
        return idx_to_coordinates,np.array(points)
    else:
        return None 
    
def get_eye_center(left_eye,right_eye):
    '''
    return:
        ndarray:
        left_eye_center,
        right_eye_center,
        eyes_center
    '''

    leftEyeCenter = left_eye.mean(axis=0)
    rightEyeCenter = right_eye.mean(axis=0)
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    return leftEyeCenter,rightEyeCenter, np.array(eyesCenter,dtype=np.float16)

def get_angle(pts1,pts2):
    dY = pts1[1] - pts2[1]
    dX = pts1[0] - pts2[0]
    angle = np.degrees(np.arctan2(dY, dX))
    if dX < 0:
        angle -= 180
    return angle

def align(img,idx_to_coordinates,scale,show= False):
    connections = [mpFaceMesh.FACEMESH_LEFT_EYE,mpFaceMesh.FACEMESH_RIGHT_EYE]
    left_eye = get_connection_points(idx_to_coordinates,connections[0])
    right_eye = get_connection_points(idx_to_coordinates,connections[1])

    leftEyeCenter, rightEyeCenter, eyesCenter = get_eye_center(left_eye,right_eye)
    angle = get_angle(rightEyeCenter,leftEyeCenter)

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    out_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                            flags=cv2.INTER_CUBIC)
    # out_img = out_img[landmarks[:, 1].min():landmarks[:, 1].max(),
        #             landmarks[:, 0].min():landmarks[:, 0].max()]
    if show:
        plt.imshow(out_img)
    return out_img

