# 🍵 Matcha — Multi-Modal Perception for End-to-End Reinforcement Learning in Balancing Robots

> *"Balancing, but make it intelligent (and a little bit cute)."* ☕🤖

---

## 🌱 Overview

**Matcha** is our experimental balancing robot project from **Fulbright University Vietnam**, exploring how *reinforcement learning* and *multi-modal perception* can enable robots to self-balance using vision and inertial sensing.

Our current research direction is:
> **Multi-Modal Perception for End-to-End Reinforcement Learning in Balancing Robots:  
A Comparison of Vision-Only and Vision+IMU Policies**

We aim to study how adding different sensory inputs (camera, IMU) affects policy robustness, stability, and real-world transfer in reinforcement-learning-based balance control.

---

## 🧩 Research Phases

| Phase | Focus | Description |
|:--:|:--|:--|
| **v1 — PPO Baseline** | State-based RL | Trained on pitch, pitch rate, and velocity — stable control achieved. |
| **v2 — Reward Tuned PPO** | Improved shaping | Adjusted reward weights to enhance stability and smooth motion. |
| **v3 — Vision-Only RL** | End-to-End Vision | Replace explicit states with CNN encoder on simulated camera feed. |
| **v4 — Vision + IMU Fusion** | Multi-Modal Learning | Combine visual and inertial features for robustness under uncertainty. |
| **v5 — Real-World Transfer** | Hardware Validation | Deploy learned policy on physical Matcha robot. |

Each version is trained, evaluated, and logged independently under the `/logs_ppo` directory for reproducibility.

---

## 🧠 Model Architecture (Simplified)

    [Camera RGB] ──┐
                    ├──▶ [Vision Encoder (CNN)] ─┐
    [IMU Sensor] ───┘                            │
                                                 ▼
                                           [Feature Fusion]
                                                 ▼
                                           [PPO Policy Net]
                                                 ▼
                                           [Motor Commands]

---

## 👩‍🔬 Team Matcha

| Member | Role | Focus Area |
|:--|:--|:--|
| **Đinh Hồng Ngọc** | Project Manager & AI / CV Engineer | Leads PPO & End-to-End RL pipelines, reward tuning, perception fusion. Also coordinates research documentation and writing.|
| **Phương Quỳnh** | Mechanical & Embedded Systems Engineer | Designs robot hardware, integrates IMU + camera, builds simulation-to-real bridge. |
| **Alex** | Research Assistant & Software Support | Runs experiments, manages logs, visualizes data, and supports evaluation pipeline. |
| **Prof. Dương Phùng** | 🎓 Research Advisor (Fulbright University) | Supervises research direction, provides guidance on RL architecture and methodology. |

---

## 📊 Current Status

- ✅ **PPO Baseline (v1)** — Stable up to *12.4 s* average survival.
- ✅ **Reward Tuned PPO (v2)** — Stable up to *14 s*, smooth control, mean reward ≈ 1965.
- 🚧 **Vision RL (v3)** — Preparing simulation camera feed, CNN encoder setup.
- 🧪 **Fusion RL (v4)** — Planned: joint embedding of vision + IMU for robust policy learning.
- 🔧 **Hardware Integration** — Physical prototype under assembly; IMU streaming functional.

---

## 🧾 Reproducibility Notes

All experiments follow consistent setup:
- **URDF**: `hardware/balance_robot.urdf`
- **Simulator**: PyBullet (240 Hz)
- **Algorithm**: PPO (Stable-Baselines3)
- **Reward Function (v2)**:

- **Torque limit**: ±2 N·m  
- **Pitch limit**: 35°  
- **Max episode steps**: 1500  
- **Random seeds**: [42, 77, 99]

---

## 📚 Roadmap

- [x] Implement PPO state-based baseline  
- [x] Tune reward & verify stability  
- [ ] Add simulated RGB camera input  
- [ ] Implement Vision-Only PPO policy  
- [ ] Integrate IMU for multi-modal fusion  
- [ ] Validate real-world transfer  
- [ ] Prepare poster & paper for submission ✨  

---

## 🧡 Acknowledgement

This project is supported by the **Fulbright Robotics Lab** and supervised by **Prof. Dương Phùng**.  
Matcha is brewed with care ☕, code, and a little chaos — but mostly learning.

---

> “Every time Matcha falls, it learns a little better how to stand.”  
> — *Team Matcha, 2025*
