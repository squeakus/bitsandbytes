# WS server that sends messages at random intervals

import asyncio
import datetime
import random
import websockets

async def people_count(websocket, path):
    while True:
        count = random.randint(0,5)
        await websocket.send(str(count))
        await asyncio.sleep(random.randint(5,15))

start_server = websockets.serve(people_count, "127.0.0.1", 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
