from pythonosc.udp_client import SimpleUDPClient

class PdSender:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.client = SimpleUDPClient(ip, port)

    def send_selection(self, setting, option):
        print(f"→ Sending to Pure Data: /select {setting} {option}")
        self.client.send_message("/select", [setting, option])
