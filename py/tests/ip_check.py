#!/usr/bin/env python3
"""
Script to check current IP configuration of Luxonis Oak-D cameras
"""
import depthai as dai

print("Scanning for DepthAI devices...")
print("=" * 60)

# Find all available devices
devices = dai.Device.getAllAvailableDevices()

if not devices:
    print("No devices found!")
    print("\nTroubleshooting:")
    print("1. Ensure cameras are powered on")
    print("2. Check POE connection")
    print("3. Try disconnecting one camera at a time")
else:
    print(f"Found {len(devices)} device(s):\n")

    for i, device_info in enumerate(devices):
        print(f"Device #{i+1}:")
        print(f"  Name: {device_info.name}")
        print(f"  MxID: {device_info.mxid}")
        print(f"  State: {device_info.state}")
        print(f"  Protocol: {device_info.protocol}")

        # Try to get bootloader info
        try:
            with dai.DeviceBootloader(device_info) as bl:
                print(f"  Bootloader Version: {bl.getVersion()}")

                # Try to read config
                if bl.isUserBootloaderSupported():
                    config = bl.readConfig()
                    if config:
                        print(f"  Configuration found:")
                        # Note: The config object may not expose IP directly
                        # You may need to connect to read network settings
                    else:
                        print(f"  No custom configuration set (using defaults)")
        except Exception as e:
            print(f"  Could not read bootloader info: {e}")

        print()

print("=" * 60)
print("\nTo change IP one camera at a time:")
print("1. Disconnect one camera")
print("2. Run ip_edit.py to configure the connected camera")
print("3. Reconnect the configured camera")
print("4. Disconnect it and connect the other camera")
print("5. Run ip_edit.py again for the second camera")
