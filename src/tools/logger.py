from datetime import datetime

class Logger:

    @staticmethod
    def info(msg):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} {msg}")

    @staticmethod
    def warn(msg):
        print(f"[WARN] {datetime.now().strftime('%H:%M:%S')} {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} {msg}")