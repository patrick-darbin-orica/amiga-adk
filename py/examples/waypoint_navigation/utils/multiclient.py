import argparse
import asyncio
from pathlib import Path

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_service_pb2 import SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file

class MultiClientSubscriber:
    """Example of subscribing to events from multiple clients."""

    def __init__(self, service_config: EventServiceConfigList) -> None:
        """Initialize the multi-client subscriber.

        Args:
            service_config: The service config.
        """
        self.service_config = service_config
        self.clients: dict[str, EventClient] = {}

        # populate the clients
        config: EventServiceConfig
        for config in self.service_config.configs:
            if not config.port:
                self.subscriptions = config.subscriptions
                continue
            self.clients[config.name] = EventClient(config)

    async def _subscribe(self, subscription: SubscribeRequest) -> None:
        # the client name is the last part of the query
        client_name: str = subscription.uri.query.split("=")[-1]
        client: EventClient = self.clients[client_name]
        # subscribe to the event
        # NOTE: set decode to True to decode the message
        async for event, message in client.subscribe(subscription, decode=False):
            # decode the message type
            message_type = event.uri.query.split("&")[0].split("=")[-1]
            print(f"Received event from {client_name}{event.uri.path}: {message_type}")

    async def run(self) -> None:
        # start the subscribe routines
        tasks: list[asyncio.Task] = []
        for subscription in self.subscriptions:
            tasks.append(asyncio.create_task(self._subscribe(subscription)))
        # wait for the subscribe routines to finish
        await asyncio.gather(*tasks)