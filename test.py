import cv2



##################################
def test_0(data):
    print(data)

def test_webcame_1(url):
    video = cv2.VideoCapture(url)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = video.read()

        # Convert the frame to a QImage
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
##################################
modules = [test_0,test_webcame_1]
def test_case(test_id,test_data):
    print(f"test code [{test_id}]")
    modules[test_id](test_data)
##################################


if __name__ == "__main__":
    test_case(0,'asd')
    test_case(1,'http://192.168.0.33:4747/video')

##################################