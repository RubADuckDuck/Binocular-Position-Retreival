import asyncio
import websockets
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

async def send_message(websocket):
    while True:
        try:
            await websocket.send('a')
            await asyncio.sleep(1 / 30)  # Sleep for approximately 33ms to send 30 times per second
        except websockets.ConnectionClosed as e:
            logging.error(f"Connection closed: {e}")
            break

async def connect_and_send(uri):
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                await send_message(websocket)
        except (websockets.ConnectionClosed, websockets.InvalidURI, websockets.InvalidHandshake) as e:
            logging.error(f"WebSocket error: {e}")
            await asyncio.sleep(5)  # Wait before trying to reconnect

if __name__ == "__main__":
    asyncio.run(connect_and_send("ws://localhost:12345"))
