import cv2
from pygame import mixer  

mixer.init()

sound_object_detected = mixer.Sound("detected.mp3")  
sound_winner = mixer.Sound("winner.mp3")  


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

classifiers = {
    "Objeto1": cv2.CascadeClassifier('cascade.xml'),
    "Objeto2": cv2.CascadeClassifier('cascade1.xml')
}

detected_objects = set()  
required_objects = set(classifiers.keys())  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for object_name, classifier in classifiers.items():
        objects = classifier.detectMultiScale(
            gray,
            scaleFactor=5,
            minNeighbors=91,
            minSize=(70, 78)
        )
        
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, object_name, (x, y - 10), 2, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if object_name not in detected_objects:
                detected_objects.add(object_name)
                sound_object_detected.play()  

    if detected_objects == required_objects:
        cv2.putText(frame, "GANADOR!", (50, 50), 2, 1, (0, 0, 255), 3, cv2.LINE_AA)
        sound_winner.play()  # Reproducir sonido de ganador

    cv2.imshow('Detección de Objetos', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
