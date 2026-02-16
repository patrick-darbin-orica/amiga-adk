#!/usr/bin/env python3
"""
List all available DepthAI devices and show TARGET_DEVICE values
"""
import depthai as dai
from typing import List

print('=' * 70)
print('AVAILABLE DEPTHAI DEVICES')
print('=' * 70)
print()

infos: List[dai.DeviceInfo] = dai.Device.getAllAvailableDevices()

if len(infos) == 0:
    print("No devices found.")
    exit(-1)

print(f"Found {len(infos)} device(s):\n")

for i, info in enumerate(infos):
    state = str(info.state).split('X_LINK_')[1]
    print(f"[{i}] {info.name} (DeviceID: {info.deviceId}, State: {state})")

print()
print('=' * 70)
print('TARGET_DEVICE CONFIGURATION')
print('=' * 70)
print()

for info in infos:
    print(f'TARGET_DEVICE = "{info.deviceId}"  # {info.name}')

print()

