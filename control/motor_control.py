"""GPIO motor driver and Action->(left,right) adapter; the only file that touches motor/sonar GPIO."""
import threading
import time

from control.state_machine import Action

PWM_HZ = 1000
FAILSAFE_S = 0.5          # cut motors if no command within this window

BASE_SPEED = 140          # forward PWM, 0..255 (motors stall below ~60)
SLOW_SPEED = 90
TURN_DELTA = 60           # lane-keeping arc: differential between sides
CHANGE_DELTA = 110        # lane-change arc: larger differential


class Wheel:
    """One L298N channel: PWM enable pin plus two direction pins."""

    def __init__(self, en, in1, in2, inv=False):
        from gpiozero import DigitalOutputDevice, PWMOutputDevice

        self.en = PWMOutputDevice(en, frequency=PWM_HZ)
        self.in1 = DigitalOutputDevice(in1)
        self.in2 = DigitalOutputDevice(in2)
        self.inv = inv

    def set(self, v):
        fwd = (v >= 0) ^ self.inv
        self.in1.value = fwd
        self.in2.value = not fwd
        self.en.value = min(abs(int(v)), 255) / 255.0

    def stop(self):
        self.en.off()
        self.in1.off()
        self.in2.off()


class Motors:
    """Left/right tank drive over 4 wheels with a background failsafe watchdog."""

    def __init__(self):
        # BCM pins and inv flags from vanguard-rover docs/wiring_pi3.svg (same chassis).
        self.bl = Wheel(12, 5, 6, inv=False)
        self.fl = Wheel(13, 16, 26, inv=False)
        self.fr = Wheel(18, 23, 24, inv=True)
        self.br = Wheel(19, 20, 21, inv=True)
        self._all = (self.bl, self.fl, self.fr, self.br)
        self._lock = threading.Lock()
        self._l = self._r = 0
        self._last = time.monotonic()
        self._run = True
        threading.Thread(target=self._watch, daemon=True).start()

    def drive(self, l, r):
        l = max(-255, min(255, int(l)))
        r = max(-255, min(255, int(r)))
        with self._lock:
            self._l, self._r = l, r
            self._last = time.monotonic()
            self.bl.set(l)
            self.fl.set(l)
            self.fr.set(r)
            self.br.set(r)

    def stop(self):
        with self._lock:
            self._l = self._r = 0
            for w in self._all:
                w.stop()

    def _watch(self):
        while self._run:
            time.sleep(0.05)
            with self._lock:
                stale = time.monotonic() - self._last > FAILSAFE_S
                if (self._l or self._r) and stale:
                    self._l = self._r = 0
                    for w in self._all:
                        w.stop()

    def close(self):
        self._run = False
        self.stop()


class Ultrasonic:
    """HC-SR04 forward distance sensor; returns centimetres."""

    def __init__(self, echo=27, trigger=17, max_distance_m=2.0):
        from gpiozero import DistanceSensor

        self.sensor = DistanceSensor(echo=echo, trigger=trigger, max_distance=max_distance_m)

    def read_cm(self):
        return self.sensor.distance * 100

    def close(self):
        self.sensor.close()


def action_to_drive(action: Action, base: int = BASE_SPEED) -> tuple[int, int]:
    """Map a state-machine Action to tank-drive (left, right) PWM values."""
    if action is Action.STOP:
        return 0, 0
    if action is Action.SLOW:
        return SLOW_SPEED, SLOW_SPEED
    if action is Action.FORWARD:
        return base, base
    if action is Action.TURN_LEFT:
        return base - TURN_DELTA, base + TURN_DELTA
    if action is Action.TURN_RIGHT:
        return base + TURN_DELTA, base - TURN_DELTA
    if action is Action.CHANGE_LEFT:
        return base - CHANGE_DELTA, base + CHANGE_DELTA
    if action is Action.CHANGE_RIGHT:
        return base + CHANGE_DELTA, base - CHANGE_DELTA
    return 0, 0


def _bringup():
    """Drive each wheel forward then reverse to verify wiring and inv flags."""
    m = Motors()
    try:
        for name, w in (("BL", m.bl), ("FL", m.fl), ("FR", m.fr), ("BR", m.br)):
            input(f"press Enter to test {name}...")
            print(f"{name} forward"); w.set(120); time.sleep(1.5)
            print(f"{name} reverse"); w.set(-120); time.sleep(1.5)
            w.stop()
    finally:
        m.close()
        print("done")


if __name__ == "__main__":
    _bringup()
