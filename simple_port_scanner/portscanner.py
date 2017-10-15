#!usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import subprocess
import sys
from datetime import datetime


def prepare():
    # Clear the screen
    subprocess.call('clear', shell=True)
    # Set the name of the remote server
    server_name = 'localhost'
    # Get the ip of the server
    server_ip = socket.gethostbyname(server_name)
    # Print the banner
    print("-"*60)
    print("Please wait, scanning server ", server_ip)
    print("-"*60)
    # Store the start time
    t1 = datetime.now()
    # Set the port range
    rng = range(100)
    try:
        for port in rng:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((server_ip, port))
            if result == 0:
                print("Port {}: OPEN".format(port))
            else:
                print("Port {}: CLOSED".format(port))
            sock.close()
    except KeyboardInterrupt:
        print("User interrupt")
        sys.exit()
    except socket.gaierror:
        print("Host can't be resolved")
        sys.exit()
    except socket.error:
        print("Couldn't connect")
        sys.exit()

    t2 = datetime.now()
    total = t2 - t1
    print("Scan completed in: ", total)

if __name__ == "__main__":
    prepare()

