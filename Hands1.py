import cv2
import mediapipe as mp

video = cv2.VideoCapture(1) #abre a webcan 1

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:

    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos=[]
    if handsPoints:
       for points in handsPoints:
        #print(points)
        mpDraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)
        for id, cord in enumerate(points.landmark):
            cx, cy = int(cord.x * w), int(cord.y * h)

            pontos.append((cx, cy))

        dedos = [8, 12, 16, 20]
        contador = 0
        if pontos:
            if pontos[4][0] < pontos[3][0]:
                contador += 1

            for x in dedos:
                if pontos[x][1] < pontos[x - 2][1]:
                    contador += 1

        cv2.putText(img, str(contador), (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (1, 1, 1), 5)

    cv2.imshow("Imagem", img)

    #encerra o programa apertando "esc"
    cv2.waitKey(1)
    if cv2.waitKey(5) == 27:
        break
video.release()
cv2.destroyAllWindows()
