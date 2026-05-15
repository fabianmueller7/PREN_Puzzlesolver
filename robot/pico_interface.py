# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:21:25 2026

@author: danie
"""

import serial
import json
import threading
import time


class PicoInterface:
    def __init__(self, port="COM22", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # allow Pico reset

        self.running = True
        self.lock = threading.Lock()

        # response handling
        self.response = None
        self.response_event = threading.Event()

        # event callbacks
        self.event_handlers = {}

        # start listener thread
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    # ------------------------
    # Internal listener thread
    # ------------------------
    def _listen(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                if not line:
                    continue

                msg = json.loads(line)

                if msg.get("type") == "response":
                    self.response = msg
                    self.response_event.set()

                elif msg.get("type") == "event":
                    self._handle_event(msg)

            except Exception as e:
                print("Error:", e)

    # ------------------------
    # Event system
    # ------------------------
    def _handle_event(self, msg):
        name = msg.get("name")
        if name in self.event_handlers:
            for callback in self.event_handlers[name]:
                callback(msg)

    def on_event(self, event_name, callback):
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(callback)

    # ------------------------
    # RPC call
    # ------------------------
    def call(self, cmd, **kwargs):
        request = {
            "type": "call",
            "cmd": cmd,
            **kwargs
        }
        print(request)

        with self.lock:
            self.response_event.clear()
            self.ser.write((json.dumps(request) + "\n").encode())

            if not self.response_event.wait(timeout=20):
                raise TimeoutError("No response from Pico")

            return self.response

    # ------------------------
    # High-level API
    # ------------------------
    def led_on(self):
        return self.call("led_on")

    def led_off(self):
        return self.call("led_off")

    def indicator_on(self):
        return self.call("indicator_on")

    def indicator_off(self):
        return self.call("indicator_off")

    def vacuum_pump_on(self):
        return self.call("vacuum_pump_on")

    def vacuum_pump_off(self):
        return self.call("vacuum_pump_off")

    def read_adc(self, pin):
        return self.call("read_adc", pin=pin)

    def go_to(self, xCoords, yCoords):
        return self.call("go_to", x=xCoords, y=yCoords)

    def go_to_z(self, zCoords):
        return self.call("go_to_z", z=zCoords)

    def gripper_rotate(self, aCoords):
        return self.call("gripper_rotate", a=aCoords)

    def home_x(self):
        return self.call("home_x")

    def home_y(self):
        return self.call("home_y")

    def home_z(self):
        return self.call("home_z")

    def reset_rotation(self):
        return self.call("reset_rotation")

    def gpo1_on(self):
        return self.call("gpo1_on")

    def gpo1_off(self):
        return self.call("gpo1_off")

    def gpo2_on(self):
        return self.call("gpo2_on")

    def gpo2_off(self):
        return self.call("gpo2_off")

    def gpo3_on(self):
        return self.call("gpo3_on")

    def gpo3_off(self):
        return self.call("gpo3_off")

    def gpo4_on(self):
        return self.call("gpo4_on")

    def gpo4_off(self):
        return self.call("gpo4_off")

    def gpo5_on(self):
        return self.call("gpo5_on")

    def gpo5_off(self):
        return self.call("gpo5_off")

    def gripper_on(self):
        return self.call("gripper_on")

    def gripper_off(self):
        return self.call("gripper_off")

    def gripper_down(self):
        return self.call("gripper_down")

    def gripper_up(self):
        return self.call("gripper_up")

    def motors_enable(self):
        return self.call("motors_enable")

    def motors_disable(self):
        return self.call("motors_disable")


    # ------------------------
    # Cleanup
    # ------------------------
    def close(self):
        self.running = False
        self.thread.join()
        self.ser.close()