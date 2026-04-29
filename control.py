# control.py
import threading

# Event untuk memulai proses scan dari frontend
scan_requested = False
scan_lock = threading.Lock()

def request_scan():
    global scan_requested
    with scan_lock:
        scan_requested = True

def consume_scan_request():
    global scan_requested
    with scan_lock:
        if scan_requested:
            scan_requested = False
            return True
        return False
    
import queue

system_state = {"STATE": "IDLE", "recognized_name": None}
status_queue = queue.Queue()

def reset_system_state():
    global system_state
    system_state["STATE"] = "IDLE"
    system_state["recognized_name"] = None
    status_queue.put({"status": "IDLE", "message": "Siap untuk scan baru"})