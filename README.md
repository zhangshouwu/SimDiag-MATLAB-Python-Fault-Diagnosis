# SimDiag V3.7 — MATLAB/Simulink + Python Fault Diagnosis Prototype

**Bilingual desktop demo for closed-loop actuator loss-of-effectiveness (LOE) studies.**

SimDiag V3.7 connects a Python/Tkinter GUI to a MATLAB/Simulink closed-loop PID model through MATLAB Engine for Python. It runs healthy and fault-injected simulations, extracts logged signals, compares responses, and generates plots, JSON metrics, and a text report.

## Features

- Real Simulink execution via MATLAB Engine for Python (MATLAB/Simulink license required).
- Configurable actuator LOE magnitude, onset time, reference step and plant denominator.
- Healthy-versus-fault response and residual plots.
- Offline paired healthy-reference calibration for a robustness benchmark with measurement noise and plant mismatch.
- Causal, sample-by-sample RLS **offline replay** of LOE estimates (not live embedded detection).
- Bilingual Chinese/English desktop UI and exportable analysis files.

## Requirements

- Windows with MATLAB + Simulink and MATLAB Engine for Python installed and compatible with your Python version.
- Python packages: `numpy`, `scipy`, `matplotlib`, `Pillow`.
- Tkinter (usually bundled with the standard Windows Python installer).

## Run

```powershell
python -m pip install -r requirements.txt
python app_v32.py
```

Select the included `model/SimDiag_PID_V14.slx` if not detected automatically. Configure the study and start the run. Outputs are saved under `reports_v37/`.

**Important:** The MATLAB Engine Python package is installed separately according to the instructions for your MATLAB release; it is not included in `requirements.txt`.

## Output artifacts

- `comparison.png` — healthy versus LOE response
- `residual.png` — residual visualization
- `loe_estimate.png` — causal RLS replay visualization
- `robustness.png` — robustness benchmark visualization
- `metrics.json` — structured results
- `SimDiag_AI_Report.txt` — bilingual report

## Scope and limitations

This is an engineering **simulation prototype**, not a production-ready safety monitor or validated real-time diagnostic product. Noise is applied to the **post-simulation diagnostic measurement channel**, not to the PID feedback path. The paired healthy-reference benchmark uses an offline reference; it is **not standalone online fault detection**. RLS estimates are computed in offline causal replay from logged data. The included Simulink model is a demonstration plant, not a validated industrial motor/actuator digital twin. No independent hardware validation is claimed.

The application is named V3.7, while the retained internal script filenames are `app_v32.py` and `sim_worker_v32.py` for compatibility with the original working package.

## Technology

Python, Tkinter, NumPy, SciPy, Matplotlib, MATLAB Engine for Python, MATLAB/Simulink.

## 中文简介

SimDiag V3.7 是 MATLAB/Simulink 与 Python 联合开发的闭环执行器效率下降（LOE）故障仿真与诊断演示程序。支持健康/故障工况对比、残差分析、离线配对健康参考鲁棒性测试、因果递推 RLS 离线回放以及报告导出。**仅为仿真原型，未宣称完成工业现场验证或实时在线部署。**
