import socket
import struct


def ip_to_int(ip):
    """IPv4 주소를 32비트 정수로 변환하는 함수입니다."""
    return struct.unpack("!I", socket.inet_aton(ip))[0]


__all__ = ["ip_to_int"]
