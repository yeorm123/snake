"""웹캠 실시간 객체 추적 (Ultralytics YOLO26 + ByteTrack)

설치:  pip install -U ultralytics opencv-python
실행:  python webcam_track.py                   (기본: yolo26n.pt, 카메라 0번)
       python webcam_track.py --tracker botsort.yaml
종료:  q 또는 ESC
"""
import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 webcam object tracking")
    parser.add_argument("--model", default="yolo26n.pt", help="yolo26n/s/m/l/x.pt")
    parser.add_argument("--cam", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml 또는 botsort.yaml")
    parser.add_argument("--trail", type=int, default=30, help="이동 경로를 그릴 프레임 수")
    args = parser.parse_args()

    model = YOLO(args.model)  # 처음 실행 시 가중치 자동 다운로드

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # Windows에서 빠른 카메라 오픈
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"웹캠 {args.cam}번을 열 수 없습니다.")

    trails = defaultdict(lambda: deque(maxlen=args.trail))  # ID별 지나온 위치
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        # persist=True: 이전 프레임의 추적 정보를 기억해서 같은 물체에 같은 ID를 유지
        results = model.track(frame, persist=True, tracker=args.tracker,
                              conf=args.conf, imgsz=args.imgsz, verbose=False)
        r = results[0]
        annotated = r.plot()

        # 각 물체의 이동 경로(꼬리) 그리기
        if r.boxes.id is not None:
            for (x, y, w, h), tid in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                trails[tid].append((int(x), int(y)))
                pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [pts], False, (0, 255, 255), 2)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        n = 0 if r.boxes.id is None else len(r.boxes.id)
        cv2.putText(annotated, f"FPS: {fps:.1f}  Tracked: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO26 Webcam Tracking", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
