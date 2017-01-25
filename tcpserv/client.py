import socket
import json
import time
data = {'message':'hello world!', 'test':123.4}

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
for i in range(10):
    s.connect(('127.0.0.1', 9999))
    time.sleep(2)
    s.send(json.dumps(data))
    result = json.loads(s.recv(1024))
    print result
    s.close()
