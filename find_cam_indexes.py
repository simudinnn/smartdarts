import cv2

from camera_capture_config import force_capture_settings

print("Pretraga dostupnih kamera...\n")

found = False

for i in range(10):
    print(f"Checking indexes {i}...", end=" ")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    force_capture_settings(cap)

    if not cap.isOpened():
        print("Not available")
        continue

    ret, frame = cap.read()

    if ret:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"OK | Resolution: {w}x{h} | FPS: {fps:.1f}")
        found = True

        cv2.imshow(f"Camera {i}", frame)
        print("Press any button for next camera...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Open, but no image.")

    cap.release()

if not found:
    print("\nNo camera found.")

print("\nDone.")
