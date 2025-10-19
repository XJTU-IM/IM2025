import socket
import time

our_ip = "192.168.1.100"    # 比赛要求设置为固定Ip
"""
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0 up
"""
our_port = 12345
judger_ip = "192.168.1.88"
judger_port = 6666

tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_client.bind((our_ip, our_port))
tcp_client.connect((judger_ip, judger_port))
print("连接成功")

# 发送起始指令
id = bytes("xjtuwhfz", 'UTF-8').hex()
tcp_client.send(bytes.fromhex("00000000" + "00000008" + id))

time.sleep(0.2)

# send results
results = "START\n" + "Goal_ID=1;Goal_A=125.8;Goal_B=8.1;Goal_C=4\n" + "END"
            
ans_len = len(results.encode())
ans_len = '{:08x}'.format(ans_len)
ans = bytes(results, 'UTF-8')
hexstr = ans.hex()

tcp_client.send(bytes.fromhex("00000002" + ans_len + hexstr))
tcp_client.close()