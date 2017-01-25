import socket, json, time

HOST = '79.97.169.97'
PORT = 8000 # Arbitrary non-privileged port
data = {'message':'hello world!', 'test':123.4}
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print "connecting to", HOST
s.connect((HOST, PORT))

for i in range(3):
    s.send(json.dumps(data))
    result = s.recv(1024)
    print result
    time.sleep(2)
s.close()
