#!/usr/bin/env python

# WS server example that synchronizes state across clients

import asyncio
import json
import websockets

STATE = {"current": 0,
         "maximum": 5}

USERS = set()


def state_event():
    return json.dumps({"type": "state", **STATE})


def users_event():
    return json.dumps({"type": "users", "count": len(USERS)})


async def notify_state():
    if USERS:  # asyncio.wait doesn't accept an empty list
        message = state_event()
        await asyncio.wait([user.send(message) for user in USERS])


async def notify_users():
    if USERS:  # asyncio.wait doesn't accept an empty list
        message = users_event()
        await asyncio.wait([user.send(message) for user in USERS])


async def register(websocket):
    USERS.add(websocket)
    await notify_users()


async def unregister(websocket):
    USERS.remove(websocket)
    await notify_users()


async def counter(websocket, path):
    # register(websocket) sends user_event() to websocket
    await register(websocket)
    try:
        await websocket.send(state_event())
        async for message in websocket:
            data = json.loads(message)
            if data["action"] == "curminus":
                STATE["current"] -= 1
                await notify_state()
            elif data["action"] == "curplus":
                STATE["current"] += 1
                await notify_state()
            elif data["action"] == "maxminus":
                STATE["maximum"] -= 1
                await notify_state()
            elif data["action"] == "maxplus":
                STATE["maximum"] += 1
                await notify_state()
            else:
                print("unsupported event: ", data)

            # no negative values
            if STATE["maximum"] < 1:
                STATE["maximum"] = 1
            if STATE["current"] < 0:
                STATE["current"] = 0

    finally:
        await unregister(websocket)


start_server = websockets.serve(counter, "localhost", 6789)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()