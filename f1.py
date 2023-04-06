import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Face Mesh 모델 초기화
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# target
image_t = cv2.imread('images2.jfif')
image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB)
# 애니메이션 캐릭터의 face mesh 좌표값
anime_face_mesh_coords = [ ... ] # 애니메이션 캐릭터의 face mesh 좌표값 리스트

# 이미지 로드
# source
image = cv2.imread('images.jfif')

# 이미지를 BGR에서 RGB로 변환
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Face Mesh 좌표값 추출

def get_face_mesh_coords(image):
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        face_mesh_coords = []
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = landmark.x * image.shape[1]
                y = landmark.y * image.shape[0]
                face_mesh_coords.append((x, y))
        return face_mesh_coords
    else:
        return None
    
face_mesh_coords = get_face_mesh_coords(image)
anime_face_mesh_coords = get_face_mesh_coords(image_t)
    
if None not in (face_mesh_coords,anime_face_mesh_coords):
    # face mesh 좌표값을 이용한 얼굴 변환
    input_coords = np.array(face_mesh_coords)
    target_coords = np.array(anime_face_mesh_coords)

    # 변환된 좌표값으로 이미지 변환
    transformation_matrix, _ = cv2.estimateAffinePartial2D(target_coords, input_coords, method=cv2.RANSAC)
    transformed_image = cv2.warpAffine(image, transformation_matrix, (image.shape[1], image.shape[0]))
    print(transformed_image.shape)
    # 이미지 출력
    cv2.imshow('Output Image', transformed_image)
    cv2.imshow('ori', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    print("None")
# Face Mesh 모델 종료
face_mesh.close()
