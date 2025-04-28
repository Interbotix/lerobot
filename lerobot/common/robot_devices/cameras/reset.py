import pyrealsense2 as rs
import time

def hardware_reset_all_d405():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("[ERROR] No RealSense devices found.")
        return

    print(f"[INFO] Found {len(devices)} RealSense devices.")

    for device in devices:
        try:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)

            if "D405" in name:
                print(f"[INFO] Performing hardware reset on D405 camera (Serial: {serial})...")
                device.hardware_reset()
            else:
                print(f"[INFO] Skipping non-D405 device: {name} (Serial: {serial})")
        except Exception as e:
            print(f"[ERROR] Failed to reset device: {e}")

    print("[INFO] Waiting 5 seconds for cameras to reboot...")
    time.sleep(5)
    print("[INFO] Reset completed.")

if __name__ == "__main__":
    hardware_reset_all_d405()
