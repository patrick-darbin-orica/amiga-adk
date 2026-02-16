import asyncio
import numpy as np
from pathlib import Path
from typing import Optional
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty

from farm_ng.filter.filter_pb2 import FilterState
from utils.track_planner import TrackBuilder

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig

from farm_ng.core.events_file_reader import proto_from_json_file

from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng_core_pybind import Isometry3F64


async def create_initial_pose(client: Optional[EventClient] = None, timeout: float = 0.5) -> Pose3F64:
    zero_tangent = np.zeros((6, 1), dtype=np.float64)
    start: Pose3F64 = Pose3F64(
        a_from_b=Isometry3F64(), frame_a="world", frame_b="robot", tangent_of_b_in_a=zero_tangent
    )
    if client is not None:
        try:
            state: FilterState = await asyncio.wait_for(
                client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
            )
            start = Pose3F64.from_proto(state.pose)
        except asyncio.TimeoutError:
            print("Timeout while getting filter state")
        except Exception as e:
            print(f"Error getting filter state {e}")

    return start


async def track_forwards(z_coordinate, client: Optional[EventClient] = None, save_track: Optional[Path] = None) -> Track:
    start: Pose3F64 = await create_initial_pose(client)

    trackbuilder = TrackBuilder(start=start)

    trackbuilder.create_straight_segment(next_frame_b="forwards", distance=z_coordinate, spacing=0.05)

    if save_track is not None:
        trackbuilder.save_track(save_track)

    return trackbuilder.track


async def set_track(service_config: EventServiceConfig, track: Track):
    print("Setting the track")
    await EventClient(service_config).request_reply("/set_track", TrackFollowRequest(track=track))


async def start(service_config: EventServiceConfig) -> None:
    print("Start moving towards the last known location of the operator")
    await EventClient(service_config).request_reply("/start", Empty())


async def run_path(service_config_path: Path, track_path: Path) -> None:
    service_config: EventServiceConfig = proto_from_json_file(service_config_path, EventServiceConfig())

    track: Track = proto_from_json_file(track_path, Track())

    await set_track(service_config, track)

    await start(service_config)


async def move_backwards(twist, client):
    # Start with the robot not moving for 1 second
    print("The Amiga will remain stationary for 1 second")
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Move the robot backwards at 0.5 m/s for 2 seconds
    print("The Amiga will now move backwards for 2 seconds")
    twist.linear_velocity_x = -0.5
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # TODO: Implement a function where the robot takes the determined Z ROI coordinate and uses
    #       it to determine how far it is from the human and move up to a set distance and stop.
    # IDEA: Is there potentially a way to set the human as a waypoint and move towards it and
    #       stop at the set distance away?

    # Stop the robot indefinitely
    print("The Amiga will now stop indefinitely")
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def coord_move_forwards(config: EventServiceConfig, client: EventClient, z_coordinate: float, save_track: Optional[Path] = None):
    """Move the robot forwards based on detected z-coordinate (forward distance in mm).

    Args:
        config: EventServiceConfig for the track_follower service
        client: EventClient to communicate with services
        z_coordinate: Forward distance in millimeters (from camera depth)
        save_track: Optional Path to save the generated track for debugging
    """
    # Convert mm to meters and apply proper geometric correction
    # (replace 1.9 with actual camera height and pitch calibration once determined)
    distance_m = (z_coordinate / 1000.0) / 1.9

    print(f"Creating straight segment: z_coordinate={z_coordinate}mm → distance={distance_m:.3f}m")

    # Create track by generating waypoints from current pose to target
    track = await track_forwards(distance_m, client, save_track)

    try:
        # Set the track on the track_follower service
        await set_track(config, track)

        # Start the robot moving along the track
        await start(config)
    except Exception as e:
        print(f"Error during track execution: {e}")

    print(f"Robot is now moving forwards for {distance_m:.3f}m")


async def move_forwards(twist, client):
    # Start with the robot not moving for 1 second
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(15.0)

    # Move the robot forwards at 0.5 m/s for 2 seconds
    twist.linear_velocity_x = 0.5
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(4.0)

    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def turn_left(twist, client):
    # Start with the robot not moving for 1 second
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Turn the robot left at 0.5 m/s for 2 seconds
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = -0.5
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def turn_right(twist, client):
    # Start with the robot not moving for 1 second
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Turn the robot right at 0.5 m/s for 2 seconds
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.5
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def stop(twist, client):
    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)