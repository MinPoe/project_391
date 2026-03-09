import argparse
import csv
import bisect
import os
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

ACKERMANN_DRIVE = """\
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""

ACKERMANN_DRIVE_STAMPED = """\
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""

def make_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    extra = {}
    extra.update(get_types_from_msg(ACKERMANN_DRIVE, 'ackermann_msgs/msg/AckermannDrive'))
    extra.update(get_types_from_msg(ACKERMANN_DRIVE_STAMPED, 'ackermann_msgs/msg/AckermannDriveStamped'))
    typestore.register(extra)
    return typestore

def extract_messages(bag_path):
    scans, drives, odoms = [], [], []
    typestore = make_typestore()
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            try:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            except Exception:
                continue
            if topic == '/scan':
                scans.append((timestamp, msg))
            elif topic == '/drive':
                drives.append((timestamp, msg))
            elif topic == '/odom':
                odoms.append((timestamp, msg))
    return scans, drives, odoms

def find_closest(target_ts, timestamps, messages):
    # binary search for nearest timestamp
    idx = bisect.bisect_left(timestamps, target_ts)
    best_idx, best_diff = idx, float('inf')
    for i in [idx - 1, idx]:
        if 0 <= i < len(timestamps):
            diff = abs(timestamps[i] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
    return messages[best_idx], best_diff

def sync_messages(scans, drives, odoms, max_time_diff_ms=50):
    max_diff_ns = max_time_diff_ms * 1_000_000
    rows = []

    # pre-extract timestamps for binary search
    drive_ts = [ts for ts, _ in drives]
    drive_msgs = [msg for _, msg in drives]
    odom_ts = [ts for ts, _ in odoms]
    odom_msgs = [msg for _, msg in odoms]

    for ts, scan_msg in scans:
        drive_msg, drive_diff = find_closest(ts, drive_ts, drive_msgs)
        odom_msg, odom_diff = find_closest(ts, odom_ts, odom_msgs)

        if drive_diff > max_diff_ns or odom_diff > max_diff_ns:
            continue

        rows.append((
            ts,
            list(scan_msg.ranges),
            drive_msg.drive.steering_angle,
            drive_msg.drive.speed,
            odom_msg.twist.twist.linear.x
        ))

    return rows

def save_csv(rows, output_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_lidar_rays = len(rows[0][1])
    headers = ["timestamp"] + [f"lidar_{i}" for i in range(num_lidar_rays)] + ["steering_angle", "speed", "odom_vx"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for ts, lidar, steering, speed, vx in rows:
            writer.writerow([ts] + lidar + [steering, speed, vx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--output', default='training_data.csv')
    parser.add_argument('--max-time-diff', type=int, default=50)
    args = parser.parse_args()

    scans, drives, odoms = extract_messages(args.bag)
    rows = sync_messages(scans, drives, odoms, max_time_diff_ms=args.max_time_diff)
    save_csv(rows, args.output)

if __name__ == '__main__':
    main()