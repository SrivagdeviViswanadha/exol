#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import picar_4wd as fc

# HSV color ranges for red (two regions in hue space)
color_dict = {
    'red': [0, 10],
    'red_2': [160, 180]
}
kernel_5 = np.ones((5, 5), np.uint8)

def color_detect_and_get_centers(img, color_name):
    resize_img = cv2.resize(img, (160, 120))
    hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([color_dict[color_name][0], 40, 40]),
                             np.array([color_dict[color_name][1], 255, 255]))
    if color_name == 'red':
        mask2 = cv2.inRange(hsv, (color_dict['red_2'][0], 40, 40), (color_dict['red_2'][1], 255, 255))
        mask = cv2.bitwise_or(mask, mask2)

    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5, iterations=1)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    sizes = []
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 3 and h >= 3:
            cx = x + w // 2
            cy = y + h // 2
            centers.append((cx, cy))
            sizes.append(h)
            cv2.rectangle(resize_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resize_img, f"#{idx}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return resize_img, mask, morph, centers, sizes

def real_forward(speed):
    fc.backward(speed)  # wiring is reversed on some models

def follow_red_object_loop(camera):
    print("🚗 Combo Mode: Follow when far, Block when close...")

    red_lost_counter = 0
    red_lost_threshold = 5

    prev_cx = None
    prev_time = time.time()

    while True:
        img = camera.capture_array()
        _, _, _, centers, sizes = color_detect_and_get_centers(img, 'red')

        if not centers:
            red_lost_counter += 1
            print(f"🔍 Red object not found — lost count: {red_lost_counter}")
            if red_lost_counter >= red_lost_threshold:
                print("🛑 Red object lost. Stopping.")
                fc.stop()
                break
            else:
                fc.stop()
                time.sleep(0.1)
                continue
        else:
            red_lost_counter = 0

        # Track object closest to center
        centers = sorted(centers, key=lambda c: abs(c[0] - 80))
        cx, cy = centers[0]
        h = sizes[0]

        now = time.time()
        dt = now - prev_time
        velocity = 0
        if prev_cx is not None and dt > 0:
            velocity = (cx - prev_cx) / dt
        lead_cx = int(cx + 0.2 * velocity)
        prev_cx, prev_time = cx, now

        offset = lead_cx - 80

        if h > 40:
            # Blocking mode (object is close)
            print(f"🛡 BLOCKING MODE — h={h}, Offset={offset}")

            if abs(offset) <= 10:
                print("✅ Aligned: stay")
                fc.stop()
            elif offset < -10:
                print("↪️ Shift RIGHT")
                fc.turn_right(40)
                time.sleep(0.2)
                fc.stop()
            else:
                print("↩️ Shift LEFT")
                fc.turn_left(40)
                time.sleep(0.2)
                fc.stop()

        else:
            # Follow mode (object is far)
            speed = min(100, int(2000 / max(1, h)))
            speed = max(30, speed)
            print(f"🚀 FOLLOW MODE — h={h}, Offset={offset}, Speed={speed:.1f}, Velocity={velocity:.2f}")

            if abs(offset) <= 20:
                print("✅ Centered: forward")
                real_forward(speed)
            elif offset < -20:
                print("↪️ Offset left: turn RIGHT")
                fc.turn_right(speed)
            else:
                print("↩️ Offset right: turn LEFT")
                fc.turn_left(speed)

            time.sleep(0.25)
            fc.stop()

        time.sleep(0.1)

# ==== MAIN ENTRY ====
with Picamera2() as camera:
    print("🎬 Starting red object tracking...")
    camera.preview_configuration.main.size = (640, 480)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()
    time.sleep(1)

    try:
        follow_red_object_loop(camera)
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
        fc.stop()

    cv2.destroyAllWindows()
    camera.close()


