"""웹캠 실시간 객체 탐지 (Ultralytics YOLO26)

설치:  pip install -U ultralytics opencv-python
실행:  python webcam_detect.py            (기본: yolo26n.pt, 카메라 0번)
       python webcam_detect.py --model yolo26s.pt --cam 1 --conf 0.4
종료:  q 또는 ESC
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 webcam object detection")
    parser.add_argument("--model", default="yolo26n.pt", help="yolo26n/s/m/l/x.pt")
    parser.add_argument("--cam", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    args = parser.parse_args()

    model = YOLO(args.model)  # 처음 실행 시 가중치 자동 다운로드

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # Windows에서 빠른 카메라 오픈
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"웹캠 {args.cam}번을 열 수 없습니다.")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated = results[0].plot()

        # 탐지된 객체 개수 / FPS 표시
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        n = len(results[0].boxes)
        cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO26 Webcam Detection", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
