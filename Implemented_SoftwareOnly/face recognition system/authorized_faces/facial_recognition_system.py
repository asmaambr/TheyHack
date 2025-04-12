import cv2
from deepface import DeepFace

# Path to the folder with authorized faces
authorized_faces_dir = "authorized_people"

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting video stream... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # DeepFace will search for matches in your folder
        result = DeepFace.find(img_path=frame, db_path=authorized_faces_dir, enforce_detection=False)

        if len(result) > 0 and len(result[0]) > 0:
            label = "Access Granted"
            color = (0, 255, 0)
        else:
            label = "Access Denied"
            color = (0, 0, 255)

    except Exception as e:
        label = "Error"
        color = (0, 0, 255)
        print("Error:", e)

    # Display the label
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Facial Recognition Access", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
