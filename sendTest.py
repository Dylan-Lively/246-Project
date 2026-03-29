import cv2

cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {w}x{h}, pixel aspect: {w/h:.4f}")