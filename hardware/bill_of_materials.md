# Bill of Materials

Hardware for the physical car. Drive is **4 DC motors driven by 2× L298N**
(one driver per side), powered from a 3S LiPo through a buck converter, with a
Raspberry Pi 3 running the pipeline. Wiring is in [`wiring.md`](wiring.md) and
[`wiring_pi3.png`](wiring_pi3.png).

| # | Component | Notes |
|---|-----------|-------|
| 1 | Raspberry Pi 3 | Runs lane detection + YOLO + control. GPIO is **3.3 V** logic. |
| 4 | DC gear motors | Two per side (front/back left, front/back right). |
| 2 | L298N dual H-bridge driver | MC1 = right side, MC2 = left side. One EN pair + one IN pair per motor. |
| 1 | Raspberry Pi Camera | CSI port, **15-pin** ribbon (the Pi 5's is 22-pin). |
| 1 | HC-SR04 ultrasonic sensor | Forward obstacle distance. **5 V** part — ECHO needs a divider to 3.3 V (see wiring). Not on the diagram yet. |
| 1 | 3S LiPo battery | 11.1 V nominal, XT60 connector. |
| 1 | Inline fuse, 3–5 A | On battery + lead (recommended). |
| 1 | PDB01 power distribution board | Battery in, fan out; the star-ground point. |
| 1 | Buck converter → 5.1 V | ≥3 A output, micro-USB pigtail to the Pi. |
| — | XT60 + silicone wire, jumper wires | XT60 direct from the LiPo; Dupont jumpers for signal. |
| — | **3D-printed chassis board** | Self-printed; carries the Pi, drivers, and power boards. |
| — | **3D-printed mounts** | Self-printed; camera, ultrasonic sensor, and motor brackets. |

## Power chain

```
3S LiPo (11.1 V, XT60) --[3–5 A fuse]--> PDB01 (star ground)
    |                                        |
    +--> L298N MC1 +12V                      +--> Buck -> 5.1 V (>=3 A) --micro-USB--> Pi 3
    +--> L298N MC2 +12V
```

- The L298N motor rails run off battery voltage via the PDB01.
- The Pi is powered **only** from the buck's micro-USB pigtail — never also
  through GPIO pins 2/4, and nothing from the L298N +5 V terminal.
- Test the buck on a meter (set 5.1 V) **before** connecting the Pi.
