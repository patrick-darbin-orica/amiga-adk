#!/usr/bin/env python3
import asyncio
from utils.canbus import trigger_dipbob

if __name__ == "__main__":
    asyncio.run(trigger_dipbob("can0"))
