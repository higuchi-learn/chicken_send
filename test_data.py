# udp_test_sender.py（コピペでOK）
import socket, json, math, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1", 5005)  # UnityのlistenPortと一致
t = 0.0
while True:
    # 画面外枠に沿ってぐるっと一周するベクトル
    x = math.cos(t)
    y = math.sin(t)
    msg = {"x": x, "y": y, "push": False}
    sock.sendto(json.dumps(msg).encode("utf-8"), addr)
    t += 0.05
    time.sleep(1/60)
