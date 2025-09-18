# main.py
# Chicken: Head orientation -> (x,y) UDP sender
# - x: mouth - eyeMid の向きベクトル f を正規化し、±45°で±1にマップ（f[0]/sin45）
# - y: (bow.y - redTop.y) / eye_dist を用いた「縦比率」
#       起動時に短時間の自動ベースライン y_base を取り、(y_base - y_raw) を ±y_full_ratio で ±1 にマップ
# - UDP: {x,y,push} を送信（push は現状 False）
# - 可視化: キーポイントと現在値/基準値をオーバレイ
#
# 実行例:
#   uv run main.py --show
# チューニング例:
#   uv run main.py --show --y_full_ratio 0.45   # 上下感度を上げる（±0.45の比率変化で±1）
#   uv run main.py --show --mirror              # 鏡像カメラ
#   uv run main.py --show --flip_x              # 左右反転が必要なら
#   uv run main.py --show --flip_y              # 上下反転が必要なら
#
# 操作:
#   [C] 現在値を y の新しい中立(ベースライン)に再設定（素早い再キャリブ）
#   [Q]/[ESC] 終了

import argparse, json, socket, threading, time, sys, os
from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO
import math

# ====== Keypoint order ======
IDX_RED_TOP   = 0
IDX_EYE_LEFT  = 1
IDX_EYE_RIGHT = 2
IDX_MOUTH     = 3
IDX_RED_LEFT  = 4
IDX_RED_RIGHT = 5
IDX_BOW       = 6
NUM_KPTS      = 7
# ============================

@dataclass
class SharedState:
    x: float = 0.0
    y: float = 0.0
    push: bool = False
    last_push_time: float = -1e9
    lock: threading.Lock = threading.Lock()

class UdpMuxSender(threading.Thread):
    def __init__(self, state, ip, port, rate_hz=60.0):
        super().__init__(daemon=True)
        self.state=state; self.addr=(ip,port)
        self.sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dt=1.0/rate_hz
    def run(self):
        while True:
            with self.state.lock:
                msg={"x":float(self.state.x),"y":float(self.state.y),"push":bool(self.state.push)}
            try:
                self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
            except Exception as e:
                print(f"[UDP send] {e}", file=sys.stderr)
            time.sleep(self.dt)

def _norm(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9: return v, 0.0
    return v/n, n

class VisionWorker(threading.Thread):
    def __init__(self, state, model_path, cam_index=0, imgsz=640, conf_thres=0.25,
                 mirror=False, show=False, show_scale=1.0,
                 device="auto", fp16=False,
                 # x: ±45°で±1（f[0]/sin45）
                 angle_for_full_x=45.0,
                 # y: 比率のフルスケール（例: 目の間隔に対する縦差の 0.5 が ±1）
                 y_full_ratio=0.50,
                 # y ベースライン自動取得
                 y_baseline_frames=30,
                 # 平滑化/デッドゾーン
                 ema_coeff=0.3, dead_zone=0.02,
                 # 反転
                 flip_x=False, flip_y=False):
        super().__init__(daemon=True)
        self.state=state
        self.model=YOLO(model_path)
        self.cam=cv2.VideoCapture(cam_index)
        self.cam.set(cv2.CAP_PROP_FPS,30)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

        self.imgsz=imgsz; self.conf=conf_thres; self.mirror=mirror
        self.show=show; self.show_scale=show_scale
        self.device=device; self.fp16=fp16

        # x 正規化用
        self.inv_sin45 = 1.0 / max(math.sin(math.radians(angle_for_full_x)), 1e-6)

        # y 正規化用（比率→±1）
        self.y_full_ratio = float(y_full_ratio)
        self.y_baseline_frames = int(y_baseline_frames)
        self._y_base = None
        self._y_base_acc = 0.0
        self._y_base_cnt = 0

        self.ema_coeff=float(ema_coeff)
        self.dead_zone=float(dead_zone)
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)

        self.xy_ema=np.array([0.0,0.0],dtype=np.float32)

    def _compute_xy(self, kxy: np.ndarray):
        # 取り出し
        eyeL, eyeR = kxy[IDX_EYE_LEFT], kxy[IDX_EYE_RIGHT]
        mouth      = kxy[IDX_MOUTH]
        redT       = kxy[IDX_RED_TOP]
        bow        = kxy[IDX_BOW]

        # --- x: mouth - eyeMid の向き → f[0]/sin45 で ±1 ---
        eyeMid = (eyeL + eyeR) * 0.5
        f_vec = mouth - eyeMid
        f, fn = _norm(f_vec)
        x = np.clip(f[0]*self.inv_sin45, -1.0, 1.0)

        # --- y: RedTop–Bow の縦差を目の間隔で割った「比率」 ---
        eye_dist = float(np.linalg.norm(eyeR - eyeL))
        if eye_dist < 1e-3:
            return None  # スケールが出ない

        # 画像yは下向き(+)なので、上:+ にするため符号に注意
        # y_raw = (bow.y - redTop.y) / eye_dist は通常「正」(下向き)
        y_raw = (bow[1] - redT[1]) / eye_dist  # 下:+ の生比率
        # この生比率から「中立 y_base」を引いて、上:+（= 比率が小さくなる方向）にしたいので:
        # y = (y_base - y_raw) / y_full_ratio
        # ※ 上を向くほど bow が上がって "y_raw" は小さくなる → y が + になる
        if self._y_base is None:
            # ベースライン収集中
            self._y_base_acc += y_raw
            self._y_base_cnt += 1
            if self._y_base_cnt >= self.y_baseline_frames:
                self._y_base = self._y_base_acc / max(self._y_base_cnt, 1)
        base = self._y_base if self._y_base is not None else y_raw  # 収集中は 0 になるように
        y = (base - y_raw) / max(self.y_full_ratio, 1e-6)
        y = float(np.clip(y, -1.0, 1.0))

        if self.flip_x: x = -x
        if self.flip_y: y = -y

        return float(x), float(y), f, y_raw, base, eye_dist

    def run(self):
        while True:
            ok, frame=self.cam.read()
            if not ok:
                time.sleep(0.01); continue
            if self.mirror:
                frame=cv2.flip(frame,1)

            res=self.model.predict(
                frame, imgsz=self.imgsz, conf=self.conf,
                verbose=False, max_det=1,
                device=None if self.device=="auto" else self.device,
                half=(self.fp16 and self.device not in ["auto","cpu"])
            )
            if not res or len(res[0].keypoints)==0:
                # 検出なし → ゆっくり原点へ
                self.xy_ema *= 0.9
                with self.state.lock:
                    self.state.x=float(self.xy_ema[0]); self.state.y=float(self.xy_ema[1])
                if self.show:
                    cv2.putText(frame,"No detection",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.imshow("chicken-debug",frame)
                    key=cv2.waitKey(1)&0xFF
                    if key in (27, ord('q'), ord('Q')): os._exit(0)
                continue

            kxy=res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)

            comp=self._compute_xy(kxy)
            if comp is None:
                # スケール不可 → 原点へ
                self.xy_ema *= 0.9
                with self.state.lock:
                    self.state.x=float(self.xy_ema[0]); self.state.y=float(self.xy_ema[1])
                if self.show:
                    cv2.putText(frame,"Unstable scale",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.imshow("chicken-debug",frame)
                    key=cv2.waitKey(1)&0xFF
                    if key in (27, ord('q'), ord('Q')): os._exit(0)
                continue

            x,y,f,y_raw,base,eye_dist = comp

            # EMA + デッドゾーン
            vec=np.array([x,y],dtype=np.float32)
            self.xy_ema=(1-self.ema_coeff)*self.xy_ema+self.ema_coeff*vec
            if abs(self.xy_ema[0])<self.dead_zone: self.xy_ema[0]=0.0
            if abs(self.xy_ema[1])<self.dead_zone: self.xy_ema[1]=0.0

            outx=float(np.clip(self.xy_ema[0],-1,1))
            outy=float(np.clip(self.xy_ema[1],-1,1))

            with self.state.lock:
                self.state.x=outx; self.state.y=outy

            if self.show:
                disp=frame.copy()
                # キーポイント描画
                colors=[(255,0,255),(0,200,255),(0,150,255),(0,255,0),(255,200,0),(255,150,0),(200,255,255)]
                for i,(u,vp) in enumerate(kxy.astype(int)):
                    cv2.circle(disp,(u,vp),5,colors[i%len(colors)],-1)
                # 情報表示
                info1=f"x={outx:+.2f}  y={outy:+.2f}"
                info2=f"y_raw={y_raw:+.3f}  y_base={base:+.3f}  eye_dist={eye_dist:.1f}  f=({f[0]:+.2f},{f[1]:+.2f})"
                cv2.putText(disp, info1, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(disp, info2, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow("chicken-debug", disp)
                key=cv2.waitKey(1)&0xFF
                if key in (27, ord('q'), ord('Q')): os._exit(0)
                if key in (ord('c'), ord('C')):
                    # 現在値を新しい中立に再設定
                    self._y_base = y_raw
                    self._y_base_acc = 0.0
                    self._y_base_cnt = 0

def parse_args():
    ap=argparse.ArgumentParser(description="Chicken head orientation -> (x,y) UDP (x: mouth-eye dir, y: redTop-bow ratio)")
    ap.add_argument("--model", type=str, default="models/best.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--udp_ip", type=str, default="127.0.0.1")
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--send_rate", type=float, default=60.0)
    ap.add_argument("--angle_for_full_x", type=float, default=45.0)
    ap.add_argument("--y_full_ratio", type=float, default=0.50)
    ap.add_argument("--y_baseline_frames", type=int, default=30)
    ap.add_argument("--ema_coeff", type=float, default=0.3)
    ap.add_argument("--dead_zone", type=float, default=0.02)
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--show_scale", type=float, default=1.0)
    return ap.parse_args()

def main():
    args=parse_args()
    state=SharedState()
    vision=VisionWorker(
        state, args.model, args.cam, args.imgsz, args.conf,
        mirror=args.mirror, show=args.show, show_scale=args.show_scale,
        device=args.device, fp16=args.fp16,
        angle_for_full_x=args.angle_for_full_x,
        y_full_ratio=args.y_full_ratio, y_baseline_frames=args.y_baseline_frames,
        ema_coeff=args.ema_coeff, dead_zone=args.dead_zone,
        flip_x=args.flip_x, flip_y=args.flip_y
    )
    sender=UdpMuxSender(state, args.udp_ip, args.udp_port, rate_hz=args.send_rate)
    vision.start(); sender.start()
    print("Running. Keys: [C]=recenter Y baseline, [Q/ESC]=quit")
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("Bye.")

if __name__=="__main__":
    main()
