import cv2
import mediapipe as mp
from deepface import DeepFace

name = "Aldar"
surname = "Gombozhapov"

face_img = cv2.imread(r"C:\Users\aldri\Desktop\ML\my_face_2\my_face.jpg")
if face_img is None:
    print("no face photo")
    exit()

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def fingers_up(lm):
    fingers = []
    tips = [4, 8, 12, 16, 20]
    fingers.append(lm.landmark[tips[0]].x < lm.landmark[tips[0] - 1].x)
    for i in range(1, 5):
        fingers.append(lm.landmark[tips[i]].y < lm.landmark[tips[i] - 2].y)
    return fingers.count(True)

def get_emotion(img):
    try:
        res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        return res[0]['dominant_emotion']
    except:
        return "emotion?"

cap = cv2.VideoCapture(0)
cv2.namedWindow("win", cv2.WINDOW_NORMAL)

with mp_face.FaceDetection(min_detection_confidence=0.6) as face_det, \
     mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_det.process(rgb)
        hands_result = hands.process(rgb)

        labels = []

        if faces.detections:
            for d in faces.detections:
                box = d.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                ww = int(box.width * w)
                hh = int(box.height * h)
                face_crop = frame[y:y+hh, x:x+ww]

                try:
                    res = DeepFace.verify(face_crop, face_img, enforce_detection=False)
                    if res['verified']:
                        print("DeepFace.verify result:", res)
                        if hands_result.multi_hand_landmarks:
                            hand = hands_result.multi_hand_landmarks[0]
                            n = fingers_up(hand)
                            print("fingers:", n)
                            if n == 1:
                                labels = [name]
                            elif n == 2:
                                labels = [surname]
                            elif n == 3:
                                emotion = get_emotion(face_crop)
                                labels = [name, surname, emotion]
                    else:
                        labels = ["unknown"]
                except:
                    print("DeepFace.verify exception:")
                    labels = ["unknown"]

                cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0,255,0), 2)
                for i, txt in enumerate(labels):
                    cv2.putText(frame, txt, (x, y + hh + 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("win", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
