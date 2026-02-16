#!/usr/bin/env python3
"""
Script to change IP address of a specific Luxonis Oak-D camera
Allows selection by device ID to avoid configuring the wrong camera
"""
import depthai as dai

def check_str(s: str):
    spl = s.split(".")
    if len(spl) != 4:
        raise ValueError(f"Entered value {s} doesn't contain 3 dots. Value has to be in the following format: '255.255.255.255'")
    for num in spl:
        if 255 < int(num):
            raise ValueError("Entered values can't be above 255!")
    return s

# Find all available devices
print("Scanning for DepthAI devices...")
print("=" * 60)
devices = dai.Device.getAllAvailableDevices()

if not devices:
    print("No devices found!")
    print("\nTroubleshooting:")
    print("1. Ensure cameras are powered on")
    print("2. Check POE connection")
    exit(1)

print(f"Found {len(devices)} device(s):\n")

# Display all available devices
for i, device_info in enumerate(devices):
    print(f"[{i}] Device:")
    print(f"    Name: {device_info.name}")

    # Try to get MxID - attribute name may vary by version
    try:
        mxid = device_info.getMxId() if hasattr(device_info, 'getMxId') else device_info.mxid
        print(f"    MxID: {mxid}")
    except (AttributeError, Exception):
        pass

    print(f"    State: {device_info.state}")
    print(f"    Protocol: {device_info.protocol}")
    print()

# Let user select which device to configure
print("=" * 60)
device_idx = int(input(f"Enter the device number to configure [0-{len(devices)-1}]: ").strip())

if device_idx < 0 or device_idx >= len(devices):
    raise ValueError(f"Invalid device number. Must be between 0 and {len(devices)-1}")

selected_device = devices[device_idx]
print()
# Get MxID safely
try:
    mxid = selected_device.getMxId() if hasattr(selected_device, 'getMxId') else selected_device.mxid
    print(f"Selected device: {selected_device.name} (MxID: {mxid})")
except (AttributeError, Exception):
    print(f"Selected device: {selected_device.name}")
print("=" * 60)
print()

# Now configure the selected device
print('"1" to set a static IPv4 address')
print('"2" to set a dynamic IPv4 address')
print('"3" to clear the config')
key = input('Enter the number: ').strip()
print('-------------------------------------')

if int(key) < 1 or 3 < int(key):
    raise ValueError("Entered value should either be '1', '2' or '3'!")

with dai.DeviceBootloader(selected_device) as bl:
    print(f"Connected to bootloader version: {bl.getVersion()}")

    if key in ['1', '2']:
        ipv4 = check_str(input("Enter IPv4: ").strip())
        mask = check_str(input("Enter IPv4 Mask: ").strip())
        gateway = check_str(input("Enter IPv4 Gateway: ").strip())
        mode = 'static' if key == '1' else 'dynamic'

        print()
        try:
            mxid = selected_device.getMxId() if hasattr(selected_device, 'getMxId') else selected_device.mxid
            print(f"Device to configure: {selected_device.name} (MxID: {mxid})")
        except (AttributeError, Exception):
            print(f"Device to configure: {selected_device.name}")
        val = input(f"Flashing {mode} IPv4 {ipv4}, mask {mask}, gateway {gateway} to the POE device. Enter 'y' to confirm. ").strip()
        if val != 'y':
            raise Exception("Flashing aborted.")

        conf = dai.DeviceBootloader.Config()
        if key == '1':
            conf.setStaticIPv4(ipv4, mask, gateway)
        elif key == '2':
            conf.setDynamicIPv4(ipv4, mask, gateway)
        (success, error) = bl.flashConfig(conf)
    elif key == '3':
        try:
            mxid = selected_device.getMxId() if hasattr(selected_device, 'getMxId') else selected_device.mxid
            val = input(f"Clear config for device {selected_device.name} (MxID: {mxid})? Enter 'y' to confirm. ").strip()
        except (AttributeError, Exception):
            val = input(f"Clear config for device {selected_device.name}? Enter 'y' to confirm. ").strip()
        if val != 'y':
            raise Exception("Operation aborted.")
        (success, error) = bl.flashConfigClear()

    if not success:
        print(f"Flashing failed: {error}")
    else:
        print(f"Flashing successful.")
        print(f"Device {selected_device.name} has been configured.")
