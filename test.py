import cv2
import numpy as np


##################################
def test_0(data):
    print(data)

def test_bounding_box_2(frame):
    height, width = frame.shape[:2]
    height = height//2

    thickness = 5  # 바운딩 박스 두께
    color = (0, 255, 0)  # 바운딩 박스 색상
    cv2.rectangle(frame, (0, 0), (thickness, height), color, thickness)  # 왼쪽 모서리
    cv2.rectangle(frame, (width - thickness, 0), (width, height), color, thickness)  # 오른쪽 모서리
    cv2.rectangle(frame, (0, 0), (width, thickness), color, thickness)  # 위쪽 모서리
    cv2.rectangle(frame, (0, height - thickness), (width, height), color, thickness)  # 아래쪽 모서리

    return frame

def test_baseline_1(url):
    video = cv2.VideoCapture(url)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = test_bounding_box_2(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


##################################
modules = [test_0,test_baseline_1,test_bounding_box_2]
def test_case(test_id,test_data):
    print(f"test code [{test_id}]")
    modules[test_id](test_data)
##################################


if __name__ == "__main__":
    # test_case(0,'asd')
    # test_case(1,'http://192.168.0.33:4747/video')

##################################

    import mediapipe as mp
    import cv2

    # 이미지 로드
    img = cv2.imread('images.jpg')
    print(img.shape)
    img= cv2.resize(img,(img.shape[1]*2,img.shape[0]*2))

    
    # FaceMesh 모델 로드
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh()

    # 랜드마크 그리기 위한 Drawing Utils 모듈 로드
    mpDraw = mp.solutions.drawing_utils

    # 이미지에서 얼굴 인식 및 랜드마크 검출
    results = faceMesh.process(img)

    # 얼굴 랜드마크 추출
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                img, 
                face_landmarks, 
                mpFaceMesh.FACEMESH_TESSELATION, # FACE_CONNECTIONS -> FACEMESH_TESSELATION
                landmark_drawing_spec=mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )
        face_landmarks1 = results.multi_face_landmarks[0]
    
    landmarks_points = []
    for i in face_landmarks1.landmark:
        landmarks_points.append((i.x,i.y))

    points = np.array(landmarks_points)
    print(points.shape)
    
    convexhull = cv2.convexHull(points)
    print(convexhull)
    
    # 이미지 출력
    cv2.imshow('Face Mesh', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()