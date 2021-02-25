import socket
host = "127.0.0.1"
port = 12345
ss = socket.socket()
ss.bind((host, port))
ss.listen()
cs, addr = ss.accept()
print(f"Connection from: {str(addr)}")
while True:
    data = cs.recv(1024).decode()
    if not data:
        break
    print(f"from connected user: {str(data)}")
    print(f"received from user: {str(data)}")
    data = input("type message: ")
    cs.send(data.encode())
cs.close()