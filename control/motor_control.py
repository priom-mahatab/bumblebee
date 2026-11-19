"""GPIO motor driver and Action->(left,right) adapter; the only file that touches motor/sonar GPIO."""
import threading
import time

from control.state_machine import Action

PWM_HZ = 1000
FAILSAFE_S = 0.5          # cut motors if no command within this window

BASE_SPEED = 110          # forward PWM, 0..255 (motors stall below ~60)
SLOW_SPEED = 80
TURN_DELTA = 60           # lane-keeping arc: differential between sides
CHANGE_DELTA = 110        # lane-change arc: larger differential
MIN_DRIVE_PWM = 60
STEER_GAIN_PWM = 250      # PWM differential per unit lane offset
CHANGE_STEER_GAIN_PWM = 320  # lane change: steering stays maxed until about halfway into the new lane
CHANGE_HEADING_WEIGHT = 0.5  # lane change: steer against the rover's angle to the lane so it doesn't swing across
MAX_STEER_DELTA = 110     # up to the base speed: the inner wheel may stop on hard turns
MAX_PWM = 255
DRIVE_TRIM = 0.10         # left side is weaker: scale left PWM up and right PWM down by this fraction
SONAR_QUEUE_LEN = 3       # median window; shorter cuts lag, 3 still holds the motor-noise false echo above 50 cm
LANE_KEEP_ACTIONS = (Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.SLOW)
LANE_CHANGE_ACTIONS = (Action.CHANGE_LEFT, Action.CHANGE_RIGHT)


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
        # BCM pins from vanguard-rover docs/wiring_pi3.svg (same chassis); inv flags verified on-car via _bringup.
        self.bl = Wheel(12, 5, 6, inv=True)
        self.fl = Wheel(13, 16, 26, inv=True)
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

    def __init__(self, echo=27, trigger=17, max_distance_m=2.0, queue_len=SONAR_QUEUE_LEN):
        from gpiozero import DistanceSensor

        self.sensor = DistanceSensor(echo=echo, trigger=trigger, max_distance=max_distance_m,
                                     queue_len=queue_len)

    def read_cm(self):
        return self.sensor.distance * 100

    def close(self):
        self.sensor.close()


def lane_keep_drive(action: Action, lane_offset: float | None, base: int = BASE_SPEED,
                    heading: float | None = None) -> tuple[int, int]:
    """Map an Action to (left, right) PWM, steering in proportion to lane_offset for lane-keeping and lane-change actions; lane changes use a stronger gain plus a heading term that damps the swing across the lane; the inner wheel may stop on hard turns."""
    steering = LANE_KEEP_ACTIONS + LANE_CHANGE_ACTIONS
    if action not in steering or lane_offset is None:
        return action_to_drive(action, base)
    speed = SLOW_SPEED if action is Action.SLOW else base
    error, gain = lane_offset, STEER_GAIN_PWM
    if action in LANE_CHANGE_ACTIONS:
        gain = CHANGE_STEER_GAIN_PWM
        if heading is not None:
            error += CHANGE_HEADING_WEIGHT * heading
    limit = min(MAX_STEER_DELTA, speed)
    delta = max(-limit, min(limit, gain * error))
    return tuple(_drive_pwm(v) for v in (round(speed + delta), round(speed - delta)))


def trim_drive(left: int, right: int, trim: float = DRIVE_TRIM) -> tuple[int, int]:
    """Correct for uneven motor sides by scaling left PWM up and right PWM down by trim, then reapplying the stall rules."""
    return _drive_pwm(round(left * (1 + trim))), _drive_pwm(round(right * (1 - trim)))


def _drive_pwm(v: int) -> int:
    """Clamp one side's PWM: held at stall PWM just below it, stopped well below it, capped at MAX_PWM."""
    if v < MIN_DRIVE_PWM / 2:
        return 0
    return min(MAX_PWM, max(MIN_DRIVE_PWM, v))


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
