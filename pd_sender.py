from pythonosc.udp_client import SimpleUDPClient

class PdSender:
    """
    Sends OSC messages to Pure Data for both track selection and effect control.
    """
    
    def __init__(self, ip="127.0.0.1", port=5005):
        """
        Initialize OSC client.
        
        Args:
            ip: Pure Data IP address (default: localhost)
            port: Pure Data listening port (default: 5005)
        """
        self.client = SimpleUDPClient(ip, port)
    
    def send_selection(self, setting, option):
        """
        Send track selection (from right hand).
        
        Args:
            setting: Instrument number (1-5)
            option: Track number (1-5)
        """
        print(f"→ Sending to Pure Data: /select {setting} {option}")
        self.client.send_message("/select", [setting, option])
    
    def send_effect(self, track, effect_type, value):
        """
        Send effect parameter (from left hand).
        
        Args:
            track: Track number to affect (1-5)
            effect_type: Effect mode (2=volume, 3=filter, 4=reverb, 5=speed)
            value: Effect parameter value (range depends on effect type)
        """
        self.client.send_message("/effect", [track, effect_type, value])