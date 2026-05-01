#!/usr/bin/env python3
import socket
from IPy import IP

class DomainIPError(Exception):
    pass

def get_ip(ip: str):
    try:
        IP(ip)
        return ip
    except ValueError: 
        try:
            return socket.gethostname(ip) 
        except:
            print()
            raise DomainIPError

def scan_port(ip: str, port: int):
    s = socket.socket()
    s.settimeout(2.0) # Optional timeout added. Remove this for more accurate results (or increase the parameter).  
    response = s.connect_ex((ip, int(port)))
    s.close()
    if response is 0:
        print("[+] Port", port, "is open on", ip + ".", end="\t")
        sock = socket.socket()
        sock.settimeout(3.0)
        try:
            sock.connect((ip, int(port)))
        except ConnectionRefusedError:
            print("[!!!] Connection refused on IP", ip +":"+str(port)+".")
            return
        try:
            banner = sock.recv(1024)
            print("[***] Banner:", str(banner.decode().strip("\n")))
            sock.close()
        except:
            print("[---] Banner not received.")
        return port
    return -1

def scan(ip: str, port_range: list):
    open_ports = []
    try:
        address = get_ip(ip)
    except DomainIPError:
        print("[-] Error: Problem with IP:", ip)
        print("[-]\tSkipping...")
        return
    if address is not ip:
        print("\nScanning", ip+"/"+address+":\n")
    else:
        print("\n[+] Scanning", ip +":\n")
    try:
        index = int(port_range[0])
        lastIndex = int(port_range[1])
    except:
        print("[-] Error: Invalid port range entered.")
        exit()
    while (index <= lastIndex):
            open_ports.append(scan_port(address, index))
            index += 1
    found_port = False
    for num in open_ports:
        if num is not -1:
            found_port = True
    if not found_port:
        print("[~] No open ports detected on", ip+".\n")

def main():
    addresses = input("[+] Enter the target(s) IP address or domain name (separate multiple targets with \",\"): ")
    ports = input("[+] Enter the target port range (example: '1-100'): ")
    if ',' in addresses:
        targets = addresses.split(',')
        try:
            ports = ports.split("-")
        except:
            print("[-] Error: Unsupported port format entered. Closing.")
            exit()
        for target in targets:
            scan(target, ports)
    else:
        try:
            ports = ports.split("-")
        except:
            print("[-] Error: Unsupported port format entered. Closing.")
            exit()
        scan(addresses, ports)

def para_main(ip_list: list, port_range: list):
    for address in ip_list:
        scan(address, port_range)

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt: # This prevents a long error message if the program is closed via ctrl+c
        print()