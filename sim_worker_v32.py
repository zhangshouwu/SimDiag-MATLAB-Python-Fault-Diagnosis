"""SimDiag V3.7 ROBUSTNESS DEMO: bilingual closed-loop LOE experiment and multi-metric analysis."""
import argparse
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.signal import lsim
# Integrated causal RLS estimator: no third Python file required.
"""Causal sampled-data LOE estimation from a logged closed-loop run.

At sample k, estimates use samples <=k only. A finite pre-trigger buffer is
replayed once when detection occurs; no future data or injected fault truth is
used. This is offline replay of a causal algorithm, NOT live Simulink execution.
"""
import numpy as np


def estimate_loe(t, pid, measured, system, trigger_time, forgetting=0.995,
                 lookback=1.5, warmup_seconds=0.4, candidate_step=0.05,
                 minimum_energy=0.05):
    t = np.asarray(t, dtype=float)
    u = np.asarray(pid, dtype=float)
    y = np.asarray(measured, dtype=float)
    if not (len(t) == len(u) == len(y) and len(t) > 3):
        raise ValueError('Inconsistent sampled signals')
    if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)):
        raise ValueError('Time must be finite and strictly increasing')
    if not 0.90 <= forgetting <= 1.0:
        raise ValueError('Invalid forgetting factor')
    dt = float(np.median(np.diff(t)))
    trigger = int(np.searchsorted(t, trigger_time))
    if trigger >= len(t):
        return {'times': [], 'values': [], 'onset': None, 'fit_rmse': None,
                'energy': 0.0, 'informative': 0, 'status': 'no samples after trigger'}
    first = max(0, trigger-int(round(lookback/dt)))
    stride = max(1, int(round(candidate_step/dt)))
    indices = np.arange(first, trigger+1, stride, dtype=int)
    if indices[-1] != trigger:
        indices = np.r_[indices, trigger]
    # Each phi[k] is a convolution of u[:k+1] with the known causal plant.
    # Computing these arrays in a batch is an implementation optimization; the
    # estimator itself never reads phi[j,k+1:] while processing sample k.
    nominal = np.asarray(lsim(system, U=u, T=t, interp=True)[1]).reshape(-1)
    phi = np.asarray([np.asarray(lsim(system, U=np.where(np.arange(len(t)) >= i, u, 0.),
                                       T=t, interp=True)[1]).reshape(-1) for i in indices])
    target = nominal-y
    count = len(indices)
    theta = np.zeros(count)
    covariance = np.full(count, 100.)
    energy = np.zeros(count)
    # Cumulative squared fit error, updated sample-by-sample. Candidate scoring
    # is computed from observations available up to the current time only.
    sse = np.zeros(count)
    times, values = [], []
    onset, fit_rmse, best_energy, informative = None, None, 0., 0
    warmup = max(1, int(round(warmup_seconds/dt)))
    for k in range(len(t)):
        x = phi[:, k]
        valid = np.abs(x) > 1e-12
        gain = covariance*x/(forgetting+covariance*x*x)
        theta += np.where(valid, gain*(target[k]-x*theta), 0.)
        covariance = np.maximum(1e-10, (covariance-gain*x*covariance)/forgetting)
        energy += x*x
        prediction_error = target[k]-np.clip(theta, 0., 1.)*x
        sse += prediction_error**2
        if k < trigger:
            continue
        times.append(float(t[k]))
        if k < trigger+warmup:
            values.append(None)
            continue
        eligible = energy >= minimum_energy
        if not np.any(eligible):
            values.append(None)
            continue
        scores = sse/(k+1) + 1e-5/(energy+1e-12)
        scores[~eligible] = np.inf
        j = int(np.argmin(scores))
        values.append(float(np.clip(theta[j], 0., 1.)))
        onset = float(t[indices[j]])
        fit_rmse = float(np.sqrt(scores[j]))
        best_energy = float(energy[j])
        informative += 1
    return {'times': times, 'values': values, 'onset': onset,
            'fit_rmse': fit_rmse, 'energy': best_energy,
            'informative': informative, 'status': 'causal buffered replay'}



def causal_ema(x, alpha=0.12):
    """One-sided low-pass filter, no future samples."""
    x = np.asarray(x, dtype=float)
    z = np.empty_like(x)
    if not len(x):
        return z
    z[0] = x[0]
    for k in range(1, len(x)):
        z[k] = alpha*x[k] + (1-alpha)*z[k-1]
    return z


def robust_detect(t, fault_residual, healthy_residual, sigma=0.0,
                  min_threshold=0.02, alpha=0.12, hold_seconds=0.10,
                  calibration_end=None):
    """Paired healthy-run calibrated detector for an OFFLINE robustness benchmark.

    Healthy residual is a separately simulated, synchronized healthy reference;
    this method is NOT a deployable unknown-plant online detector. Calibration
    uses only the healthy reference, not the injected fault or fault onset.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(fault_residual, dtype=float)
    h = np.asarray(healthy_residual, dtype=float)
    if len(t) < 4 or len(f) != len(t) or len(h) != len(t):
        raise ValueError('Invalid detector input arrays')
    dt = float(np.median(np.diff(t)))
    # Paired baseline removes repeatable healthy-model transient bias.
    # Never use the faulty trace to fit or set the threshold.
    corrected = causal_ema(f - h, alpha)
    # Conservative threshold for smoothed diagnostic-channel Gaussian noise.
    # The nominal minimum threshold is preserved as a floor.
    # For EMA white noise, sigma_out = sigma*sqrt(alpha/(2-alpha)).
    noise_floor = 5.0 * float(sigma) * np.sqrt(alpha/(2.0-alpha))
    threshold = max(float(min_threshold), noise_floor)
    required = max(3, int(np.ceil(hold_seconds/dt)))
    count = 0
    alarm = None
    for k, value in enumerate(corrected):
        count = count+1 if abs(value) > threshold else 0
        if count >= required:
            alarm = float(t[k]); break
    return {'alarm': alarm, 'threshold': threshold, 'filtered_residual': corrected,
            'hold_samples': required, 'method': 'paired healthy-reference EMA + noise-aware hold'}


def finite(value, name):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'{name}: 必须为有限数 / Must be finite')
    return x


def read_den(text):
    text = str(text).strip()

    if not (text.startswith('[') and text.endswith(']')):
        raise ValueError("分母格式错误，例如 [1 2 1]")

    content = text[1:-1].replace(',', ' ').replace(';', ' ')

    try:
        parts = [float(x) for x in content.split()]
    except ValueError as exc:
        raise ValueError("分母包含无效数值") from exc

    if len(parts) < 2:
        raise ValueError("分母至少需要两个系数")

    if not all(math.isfinite(x) for x in parts):
        raise ValueError("分母系数必须为有限数值")

    if parts[0] == 0:
        raise ValueError("分母最高次项系数不能为零")

    return parts


def signal_arrays(eng, result, signal):
    eng.workspace["sd_result_v20"] = result
    eng.workspace["sd_signal_v20"] = signal
    eng.eval("""
    sd_logs_v20 = sd_result_v20.logsout;
    if ~isa(sd_logs_v20, 'Simulink.SimulationData.Dataset')
        error('logsout is not a Dataset');
    end
    sd_element_v20 = sd_logs_v20.getElement(sd_signal_v20);
    if isempty(sd_element_v20)
        error('Requested logged signal was not found');
    end
    sd_values_v20 = sd_element_v20.Values;
    sd_payload_v20 = jsonencode(struct( ...
        'time', reshape(double(sd_values_v20.Time), 1, []), ...
        'data', reshape(double(sd_values_v20.Data), 1, [])));
    """, nargout=0)
    payload = json.loads(str(eng.workspace["sd_payload_v20"]))
    t = np.asarray(payload["time"], dtype=float).reshape(-1)
    y = np.asarray(payload["data"], dtype=float).reshape(-1)
    if len(t) < 2 or len(t) != len(y):
        raise ValueError(f"信号数据无效: {signal}")
    if not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError(f"信号包含非有限数值: {signal}")
    return t, y
def window_metrics(t, y, start, end, ref, step_time, tolerance=0.02):
    # Include exact window endpoints by interpolation, making time-integrals comparable.
    if start < t[0] or end > t[-1] or end <= start:
        return None
    inside = (t > start) & (t < end)
    tt = np.concatenate(([start], t[inside], [end]))
    yy = np.interp(tt, t, y)
    error = ref - yy
    duration = end - start
    ise = float(np.trapezoid(error ** 2, tt))
    iae = float(np.trapezoid(np.abs(error), tt))
    rmse = math.sqrt(ise / duration)
    direction = 1 if ref >= np.interp(step_time, t, y) else -1
    overshoot = max(0.0, float(np.max(direction * (yy - ref))))
    # Settling time: last excursion outside tolerance; N/A if not settled by end.
    band = tolerance * max(abs(ref), 1e-9)
    out_of_band = np.where(np.abs(error) > band)[0]
    if len(out_of_band) == 0:
        settling = 0.0
    elif out_of_band[-1] == len(tt) - 1:
        settling = None
    else:
        settling = float(tt[out_of_band[-1] + 1] - start)
    return {'rmse': rmse, 'iae': iae, 'ise': ise,
            'max_abs_error': float(np.max(np.abs(error))),
            'overshoot_absolute': overshoot, 'settling_time_from_window_start': settling,
            'samples': int(len(tt)), 'duration': duration}
def relative_change(a, b):
    return 100 * (b - a) / a if abs(a) > 1e-12 else None
def fmt(x, digits=4):
    return f'{x:.{digits}f}' if x is not None else 'N/A'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    model_path = Path(cfg['model_path']).resolve()
    if not model_path.is_file() or model_path.suffix.lower() != '.slx':
        raise ValueError('模型文件无效 / Invalid .slx model')
    model = model_path.stem
    block = str(cfg.get('block_path', 'Plant')).strip()
    profile = str(cfg.get('profile_block', 'Fault_Profile')).strip()
    signal = str(cfg.get('signal_name', 'plant_output')).strip()
    control_signal = str(cfg.get('control_signal', 'pid_output')).strip()
    if not all((block, profile, signal, control_signal)):
        raise ValueError('模块和信号不能为空 / Block and signal names required')
    normal_den = read_den(cfg.get('normal_den', '[1 2 1]'))
    fault_time = finite(cfg.get('fault_time', 3), 'Fault Time')
    loss = finite(cfg.get('fault_magnitude', 0.3), 'LOE')
    step_time = finite(cfg.get('step_time', 1), 'Step Time')
    reference = finite(cfg.get('reference', 1), 'Reference')
    stop_time = finite(cfg.get('stop_time', 10), 'Stop Time')
    if not 0 <= loss <= 1 or not 0 <= step_time < stop_time or not 0 <= fault_time < stop_time:
        raise ValueError('故障比例或时间范围无效 / Invalid LOE or time range')
    noise_sigma = finite(cfg.get('noise_sigma', 0.02), 'Noise sigma')
    mismatch_pct = finite(cfg.get('mismatch_pct', 20), 'Model mismatch percent')
    noise_seed = int(cfg.get('noise_seed', 2026))
    if not (0 <= noise_sigma <= 0.5 and -50 <= mismatch_pct <= 50):
        raise ValueError('Noise sigma must be 0..0.5; mismatch -50..50 percent')
    # Actual plant denominator scales s and s^0 coefficients while s coefficient stays fixed.
    # This changes plant dynamics rather than merely relabeling the nominal model.
    actual_den = [normal_den[0] * (1 + mismatch_pct/100),
                  normal_den[1], normal_den[2] * (1 + mismatch_pct/100)]
    out = Path(cfg['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'Worker source / 工作程序: {Path(__file__).resolve()}', flush=True)
    print(f'Output directory / 输出目录: {out}', flush=True)
    import matlab.engine
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('启动 MATLAB / Starting MATLAB...', flush=True)
    eng = matlab.engine.start_matlab()
    loaded = False
    full_plant = block if block.startswith(model + '/') else model + '/' + block.lstrip('/')
    full_profile = profile if profile.startswith(model + '/') else model + '/' + profile.lstrip('/')
    full_reference = model + '/Reference'
    original = {}
    try:
        eng.load_system(str(model_path), nargout=0)
        loaded = True
        for path, param in [(full_plant, 'Denominator'), (full_profile, 'Time'),
                            (full_profile, 'Before'), (full_profile, 'After'),
                            (full_reference, 'Time'), (full_reference, 'Before'),
                            (full_reference, 'After')]:
            original[(path, param)] = eng.get_param(path, param)
        original[(model, 'SignalLogging')] = eng.get_param(model, 'SignalLogging')
        eng.set_param(model, 'SignalLogging', 'on', nargout=0)
        

        # MATLAB Simulink block parameters must be MATLAB expression strings.
        normal_den_str = '[' + ' '.join(format(x, '.15g') for x in normal_den) + ']'
        print('设置 Plant 分母 / Denominator:', normal_den_str, flush=True)
        eng.set_param(full_plant, 'Denominator', normal_den_str, nargout=0)
        eng.set_param(full_profile, 'Time', format(fault_time, '.15g'), nargout=0)
        eng.set_param(full_reference, 'Time', format(step_time, '.15g'),
                      'Before', '0', 'After', format(reference, '.15g'), nargout=0)
        # The estimator below assumes G(s)=1/(a2*s^2+a1*s+a0), zero IC.
        # Fail explicitly when the Simulink plant does not satisfy that model.
        eng.workspace['sd_num_expr_v31'] = str(eng.get_param(full_plant, 'Numerator'))
        # Transfer Fcn blocks do not expose an InitialStates dialog parameter.
        # Inspect only the numerator; zero initial states remain an explicit
        # modeling assumption to validate against the actual simulation.
        eng.eval("sd_num_v31=eval(sd_num_expr_v31);", nargout=0)
        eng.eval("sd_num_json_v31=jsonencode(reshape(double(sd_num_v31),1,[]));", nargout=0)
        num = np.atleast_1d(np.asarray(json.loads(str(eng.workspace['sd_num_json_v31'])), dtype=float)).reshape(-1)
        if len(num) != 1 or not np.isclose(num[0], 1.0, atol=1e-10):
            raise ValueError(f'Plant numerator must be [1]; actual: {num.tolist()}')
        print('提示 / Note: Transfer Fcn initial states are assumed zero; not verified by block parameter.', flush=True)

        def simulate(before, after, label):
            print(label, flush=True)
            eng.set_param(full_profile, 'Before', str(before), 'After', str(after), nargout=0)
            eng.workspace['sd_stop_v15'] = format(stop_time, '.15g')
            eng.workspace['sd_model_v15'] = model
            eng.eval("sd_input_v15 = Simulink.SimulationInput(sd_model_v15);", nargout=0)
            eng.eval("sd_input_v15 = sd_input_v15.setModelParameter('StopTime',sd_stop_v15,'SignalLogging','on');", nargout=0)
            result = eng.sim(eng.workspace['sd_input_v15'], nargout=1)
            return signal_arrays(eng, result, signal), signal_arrays(eng, result, control_signal)

        (nt, ny), (nut, nuv) = simulate(1, 1, '正常闭环仿真 / Normal closed-loop simulation...')
        (ft, fy), (ut, uv) = simulate(1, format(1 - loss, '.15g'),
                                   '执行器效率下降仿真 / Actuator LOE closed-loop simulation...')
        print('鲁棒性工况：实际对象参数偏差 / Running mismatched plant...', flush=True)
        eng.set_param(full_plant, 'Denominator', '['+' '.join(format(v,'.15g') for v in actual_den)+']', nargout=0)
        (mt, my), (mut, muv) = simulate(1, format(1-loss,'.15g'),
                                        '模型失配闭环仿真 / Mismatched plant simulation...')
        (mht, mhy), (mhut, mhuv) = simulate(1, 1,
                                            '模型失配健康对照 / Mismatched healthy simulation...')

    finally:
        try:
            if loaded:
                for (path, param), value in original.items():
                    try:
                        eng.set_param(path, param, value, nargout=0)
                    except Exception as exc:
                        print(f'参数恢复警告 / Restore warning: {exc}', flush=True)
                eng.close_system(model, 0, nargout=0)
        finally:
            eng.quit()
    # Model-based diagnosis. Ground truth is excluded from all estimation steps.
    from scipy.signal import lsim
    if len(normal_den) != 3:
        raise ValueError('LOE estimator requires a second-order plant')
    sample_dt = finite(cfg.get('rls_dt', 0.01), 'Sample time')
    forgetting = finite(cfg.get('rls_forgetting_factor', cfg.get('loe_grid_step', 0.995)), 'RLS lambda')
    if not (0.002 <= sample_dt <= 0.1 and 0.90 <= forgetting <= 1):
        raise ValueError('Invalid sample time or forgetting factor')
    for name, t in [('normal output', nt), ('fault output', ft), ('normal PID', nut), ('fault PID', ut)]:
        if np.any(np.diff(t) <= 0):
            raise ValueError(f'{name}: timestamps must be strictly increasing')
    common_start = max(nt[0], ft[0], nut[0], ut[0])
    common_end = min(nt[-1], ft[-1], nut[-1], ut[-1], stop_time)
    start = max(step_time, fault_time)
    if common_start > 1e-8 or common_end <= start + 0.5:
        raise ValueError('Need synchronous logs from t=0 and sufficient post-fault data')
    grid = np.unique(np.r_[nt[(nt >= common_start)&(nt <= common_end)],
                            ft[(ft >= common_start)&(ft <= common_end)], start, common_end])
    normal_interp = np.interp(grid, nt, ny)
    fault_interp = np.interp(grid, ft, fy)
    comparison_residual = fault_interp - normal_interp
    ts = np.arange(0, common_end + 1e-9, sample_dt)
    ys, us = np.interp(ts, ft, fy), np.interp(ts, ut, uv)
    normal_ys, normal_us = np.interp(ts, nt, ny), np.interp(ts, nut, nuv)
    a2, a1, a0 = normal_den
    A = np.array([[0., 1.], [-a0/a2, -a1/a2]])
    B = np.array([[0.], [1./a2]])
    C = np.array([[1., 0.]])
    D = np.array([[0.]])
    system = (A, B, C, D)
    def predict(input_signal):
        # Linear interpolation matches continuous Simulink solver logs more closely
        # than the previous zero-order-hold discrete approximation.
        return np.asarray(lsim(system, U=input_signal, T=ts, interp=True)[1], dtype=float).reshape(-1)
    pred_nom = predict(us)
    healthy_prediction = predict(normal_us)
    healthy_residual = normal_ys - healthy_prediction
    healthy_model_rmse = float(np.sqrt(np.mean(healthy_residual**2)))
    healthy_model_peak = float(np.max(np.abs(healthy_residual)))
    model_tolerance = max(0.005, 0.01 * max(abs(reference), 1e-9))
    model_consistent = healthy_model_rmse <= model_tolerance
    # Detection is evaluated on the fault-run innovation only. The separate
    # healthy run calibrates a conservative threshold; it is not subtracted.
    innovation_series = ys - pred_nom
    baseline = healthy_residual
    residual_threshold = max(0.02 * max(abs(reference), 1e-9),
                             5 * float(np.sqrt(np.mean(baseline**2))),
                             2 * float(np.quantile(np.abs(baseline), 0.995)))
    threshold_mode = 'healthy-run model-error calibrated, minimum 2% reference'
    detection_rule = 'three consecutive samples above absolute threshold'
    detection_time = None
    exceed = np.abs(innovation_series) > residual_threshold
    for k in range(2, len(ts)):
        if exceed[k-2:k+1].all():
            detection_time = float(ts[k]); break
    residual_detected = detection_time is not None
    detection_delay = detection_time - fault_time if residual_detected else None
    residual = np.interp(grid, ts, innovation_series)
    post = grid >= start
    residual_post = residual[post]
    residual_rmse = math.sqrt(float(np.trapezoid(residual_post**2, grid[post]) / (common_end-start)))
    residual_peak = float(np.max(np.abs(residual_post)))
    residual_iae = float(np.trapezoid(np.abs(residual_post), grid[post]))
    # Sequential causal RLS replay. At sample k only samples <= k are used.
    # A bounded pre-trigger buffer is replayed once after the detector fires.
    estimate_times, estimate_values = [], []
    estimated_loe = estimated_onset = estimate_fit_rmse = None
    informative, excitation_energy = 0, 0.0
    confidence = 'unavailable'
    warmup = max(1, int(round(0.4/sample_dt)))
    if not model_consistent:
        confidence = 'invalid: healthy-run plant model mismatch'
        print('模型一致性失败：禁止输出LOE数值 / Model mismatch: LOE estimate disabled', flush=True)
    elif detection_time is None:
        confidence = 'unavailable: no detection trigger'
    else:
        result_rls = estimate_loe(
            ts, us, ys, system, detection_time, forgetting=forgetting,
            lookback=min(5., max(sample_dt, finite(cfg.get('onset_lookback', 1.5), 'Lookback'))),
            warmup_seconds=0.4)
        estimate_times = result_rls['times']
        estimate_values = result_rls['values']
        estimated_onset = result_rls['onset']
        estimate_fit_rmse = result_rls['fit_rmse']
        informative = result_rls['informative']
        excitation_energy = result_rls['energy']
        valid_estimates = [v for v in estimate_values if v is not None]
        if valid_estimates:
            estimated_loe = float(valid_estimates[-1])
            recent = np.asarray(valid_estimates[-max(5,int(0.5/sample_dt)):])
            variation = float(np.ptp(recent))
            confidence = ('stable causal RLS replay; independent validation required'
                          if len(recent) >= 5 and variation < 0.03 else 'transient/not converged')
        else:
            confidence = 'unavailable: insufficient excitation or short observation window'
    print(f'健康模型误差 / Healthy model RMSE: {healthy_model_rmse:.6f}', flush=True)
    print(f'LOE估计状态 / Estimator status: {confidence}', flush=True)
    false_alarm = detection_time is not None and detection_time < fault_time
    overall_normal = window_metrics(nt, ny, step_time, common_end, reference, step_time)
    overall_fault = window_metrics(ft, fy, step_time, common_end, reference, step_time)
    post_normal = window_metrics(nt, ny, start, common_end, reference, step_time)
    post_fault = window_metrics(ft, fy, start, common_end, reference, step_time)
    pre_normal = window_metrics(nt, ny, step_time, fault_time, reference, step_time) if fault_time > step_time else None
    pre_fault = window_metrics(ft, fy, step_time, fault_time, reference, step_time) if fault_time > step_time else None
    if not all((overall_normal, overall_fault, post_normal, post_fault)):
        raise ValueError('有效仿真区间不足 / Insufficient valid simulation window')
    post_change = relative_change(post_normal['rmse'], post_fault['rmse'])
    if post_change is None:
        performance = 'RMSE基准接近零 / Near-zero Baseline RMSE'
    elif post_change >= 30:
        performance = '明显RMSE恶化 / Significant RMSE Degradation'
    elif post_change >= 15:
        performance = '中度RMSE恶化 / Moderate RMSE Degradation'
    elif post_change >= 5:
        performance = '轻微RMSE恶化 / Minor RMSE Degradation'
    elif post_change <= -5:
        performance = 'RMSE降低 / RMSE Improved'
    else:
        performance = 'RMSE变化较小 / Small RMSE Change'
    detection = '检测到响应偏差 / Response Deviation Detected' if residual_detected else '未超出演示阈值 / Below Demo Threshold'
    # Robustness benchmark: paired healthy-reference calibration. Each scenario
    # uses a separately simulated healthy run of the SAME plant configuration.
    # This is an offline paired-reference demonstration, NOT deployable adaptive
    # identification under unknown mismatch. Fault truth is validation only.
    rng = np.random.default_rng(noise_seed)
    noisy_y = ys + rng.normal(0.0, noise_sigma, size=len(ts))
    noisy_healthy_y = normal_ys + rng.normal(0.0, noise_sigma, size=len(ts))
    mismatch_y = np.interp(ts, mt, my)
    mismatch_u = np.interp(ts, mut, muv)
    mismatch_healthy_y = np.interp(ts, mht, mhy)
    mismatch_healthy_u = np.interp(ts, mhut, mhuv)
    mismatch_noisy_y = mismatch_y + rng.normal(0.0, noise_sigma, size=len(ts))
    mismatch_noisy_healthy_y = mismatch_healthy_y + rng.normal(0.0, noise_sigma, size=len(ts))
    def scenario(name, yobs, upid, healthy_y, healthy_u, scenario_sigma):
        innovation = yobs - predict(upid)
        healthy_innovation = healthy_y - predict(healthy_u)
        raw_healthy_alarm = None
        raw_exceed = np.abs(healthy_innovation) > residual_threshold
        for i in range(2, len(ts)):
            if np.all(raw_exceed[i-2:i+1]):
                raw_healthy_alarm = float(ts[i]); break
        det = robust_detect(ts, innovation, healthy_innovation, sigma=scenario_sigma,
                            min_threshold=max(0.02*max(abs(reference),1e-9), 0.005))
        # An independent healthy-only diagnostic run: same measurement-noise
        # magnitude, but a new random draw, tested against calibrated baseline.
        health_validation = healthy_y + rng.normal(0, scenario_sigma, len(ts)) if scenario_sigma else healthy_y.copy()
        health_validation_innovation = health_validation - predict(healthy_u)
        health_det = robust_detect(ts, health_validation_innovation, healthy_innovation,
                                   sigma=scenario_sigma, min_threshold=max(0.02*max(abs(reference),1e-9),0.005))
        alarm = det['alarm']
        healthy_alarm = health_det['alarm']
        pre_false = alarm is not None and alarm < fault_time
        is_valid = alarm is not None and not pre_false and healthy_alarm is None
        estimated = None
        if is_valid:
            # Correct repeatable nominal-vs-actual model bias using the paired
            # healthy innovation. Equivalent adjusted measurement is y - bias.
            # Note: this requires a matched healthy reference trajectory.
            corrected_y = yobs - healthy_innovation
            fit = estimate_loe(ts, upid, corrected_y, system, alarm,
                               forgetting=forgetting, lookback=1.5,
                               warmup_seconds=0.4)
            valid = [v for v in fit['values'] if v is not None]
            if valid:
                estimated = float(valid[-1])
        return {'name': name, 'detection_time': alarm,
                'detection_delay': alarm-fault_time if is_valid else None,
                'pre_fault_false_alarm': bool(pre_false),
                'healthy_run_false_alarm': healthy_alarm is not None,
                'healthy_alarm_time': healthy_alarm,
                'raw_healthy_alarm_time': raw_healthy_alarm,
                'estimated_loe': estimated,
                'loe_error_pp': 100*abs(estimated-loss) if estimated is not None else None,
                'residual_peak': float(np.max(np.abs(innovation))),
                'healthy_residual_peak': float(np.max(np.abs(healthy_innovation))),
                'adaptive_threshold': det['threshold'],
                'hold_samples': det['hold_samples'],
                'valid_diagnosis': bool(is_valid and estimated is not None),
                'method': det['method']}
    scenarios = [
        scenario('Baseline', ys, us, normal_ys, normal_us, 0.0),
        scenario('Sensor noise (diagnostic channel)', noisy_y, us,
                 noisy_healthy_y, normal_us, noise_sigma),
        scenario('Plant parameter mismatch', mismatch_y, mismatch_u,
                 mismatch_healthy_y, mismatch_healthy_u, 0.0),
        scenario('Noise + mismatch', mismatch_noisy_y, mismatch_u,
                 mismatch_noisy_healthy_y, mismatch_healthy_u, noise_sigma),
    ]
    print('V3.7 robustness benchmark / 鲁棒性测试:', json.dumps(scenarios, ensure_ascii=False), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ['Baseline', 'Noise', 'Mismatch', 'Combined']
    xs = np.arange(4)
    for ax, field, title, unit in [
        (axes[0], 'detection_delay', 'Detection delay (valid alarms only)', 's'),
        (axes[1], 'loe_error_pp', 'LOE error (valid estimates only)', 'pp')]:
        values = [s[field] for s in scenarios]
        numeric = [float(v) for v in values if v is not None]
        ymax = max([0.2 if field=='detection_delay' else 1.0]+numeric)*1.5
        for i, (s, value) in enumerate(zip(scenarios, values)):
            invalid = s['healthy_run_false_alarm'] or s['pre_fault_false_alarm']
            if invalid:
                note = 'FALSE ALARM'
            elif value is None:
                note = 'N/A'
            else:
                note = f'{value:.3f}'
                ax.bar(i, value, color='#2676ae', width=0.7)
            ax.text(i, (value or 0)+0.03*ymax, note, ha='center', fontsize=8,
                    color='#bd4430' if invalid else '#222222')
        ax.set_xticks(xs, names, rotation=15)
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.grid(axis='y', alpha=.3)
        ax.set_axisbelow(True)
    fig.suptitle(f'SimDiag V3.7 | Paired-reference robustness benchmark | noise={noise_sigma:g}, mismatch={mismatch_pct:+g}%')
    fig.tight_layout()
    fig.savefig(out/'robustness.png', dpi=160)
    plt.close(fig)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(nt, ny, color='blue', label='Normal / 正常', linewidth=1.8)
    ax.plot(ft, fy, color='red', linestyle='--', label='Actuator LOE / 执行器效率下降', linewidth=1.8)
    ax.axvline(fault_time, color='gray', linestyle=':', label='Fault Onset / 故障开始')
    ax.set(xlabel='Time (s) / 时间 (秒)', ylabel='System Output / 系统输出', title='SimDiag V3.7 ROBUSTNESS DEMO - Closed-loop Actuator LOE')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(grid, residual, color='purple', label='Fault output - nominal predictor / 单次运行模型残差', linewidth=1.8)
    if detection_time is not None:
        ax.axvline(detection_time, color='green', linestyle='-.', label='Detection / 检测')
    ax.axhline(residual_threshold, color='orange', linestyle='--', label='Demo Threshold / 演示阈值')
    ax.axhline(-residual_threshold, color='orange', linestyle='--')
    ax.axvline(fault_time, color='gray', linestyle=':', label='Fault Onset / 故障开始')
    ax.set(xlabel='Time (s) / 时间 (秒)', ylabel='Output Residual / 输出残差', title='SimDiag V3.7 ROBUSTNESS DEMO - Response Residual')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'residual.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.axhline(loss * 100, color='blue', linestyle='--', linewidth=1.4,
               label='Injected LOE (validation only) / 注入值（仅用于验证）')
    ax.axvline(fault_time, color='gray', linestyle=':', linewidth=1.4,
               label=f'Injected onset / 注入时刻 {fault_time:.2f} s')
    if detection_time is not None:
        ax.axvline(detection_time, color='green', linestyle='-.', linewidth=1.3,
                   label=f'Detection / 检测时刻 {detection_time:.2f} s')
    valid_points = [(float(ti), float(vi)*100) for ti, vi in zip(estimate_times, estimate_values)
                    if vi is not None and np.isfinite(vi)]
    if valid_points:
        valid_t, valid_v = zip(*valid_points)
        ax.step(valid_t, valid_v, where='post', color='purple', linewidth=1.8,
                label='Causal replay estimate / 因果回放估计')
        ax.axvline(valid_t[0], color='darkorange', linestyle=':', linewidth=1.3,
                   label=f'First valid estimate / 首次有效估计 {valid_t[0]:.2f} s')
    else:
        ax.text(0.5, 0.5, 'No valid LOE estimate / 暂无有效估计',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)
    ax.set(xlabel='Time (s) / 时间 (秒)', ylabel='LOE (%) / 效率损失 (%)',
           title='SimDiag V3.7 - Causal RLS Replay LOE Estimate')
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'loe_estimate.png', dpi=160)
    plt.close(fig)
    metrics = {
        'version': '3.7-robust-detection', 'robustness_method': 'offline paired healthy-reference calibration (NOT standalone online diagnosis)', 'robustness_scenarios': scenarios, 'noise_sigma': noise_sigma, 'noise_seed': noise_seed, 'mismatch_pct': mismatch_pct, 'mismatched_actual_den': actual_den, 'noise_location': 'post-simulation diagnostic measurement only (not PID feedback)', 'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model': model, 'signal': signal, 'fault_type': 'actuator_loe',
        'fault_time': fault_time, 'fault_magnitude': loss, 'normal_den': normal_den,
        'simulation_mode': 'Simulink closed-loop actuator input fault',
        'healthy_model_rmse': healthy_model_rmse, 'healthy_model_peak': healthy_model_peak,
        'model_consistent': model_consistent, 'model_tolerance': model_tolerance,
        'normal_rmse': overall_normal['rmse'], 'fault_rmse': overall_fault['rmse'],
        'post_degradation': post_change, 'severity': performance,
        'detection': detection, 'residual_detected': residual_detected,
        'residual_threshold': residual_threshold, 'threshold_mode': threshold_mode,
        'detection_time': detection_time, 'detection_delay': detection_delay,
        'estimated_loe': estimated_loe, 'estimated_onset': estimated_onset, 'onset_method': 'causal candidate-onset bank; bounded pre-trigger buffer (not physical onset validation)',
        'estimate_times': estimate_times, 'estimate_values': estimate_values,
        'estimate_fit_rmse': estimate_fit_rmse, 'estimator_status': (confidence if estimated_loe is not None else 'insufficient excitation or no detection'), 'false_alarm': bool(false_alarm),
        'estimation_method': 'sample-by-sample causal RLS replay with bounded pre-trigger buffer',
        'estimation_limitations': 'Offline replay of a causal sample-by-sample estimator, NOT a live Simulink online detector. Single-run residual uses known nominal plant and logged PID output; zero initial conditions, sufficient excitation, accurate model, synchronized logs assumed. Threshold is demonstration-only; no independent noise validation.',
        'detection_source': 'fault-run plant output minus nominal-model prediction driven by fault-run PID command', 'detection_rule': detection_rule, 'residual_rmse': residual_rmse,
        'residual_peak': residual_peak, 'comparison_residual_peak': float(np.max(np.abs(comparison_residual))), 'residual_iae': residual_iae,
        'overall_normal': overall_normal, 'overall_fault': overall_fault,
        'pre_normal': pre_normal, 'pre_fault': pre_fault,
        'post_normal': post_normal, 'post_fault': post_fault,
        'analysis_start': start, 'analysis_end': common_end,
        'estimator_confidence': confidence, 'rls_informative_samples': informative, 'rls_excitation_energy': float(excitation_energy) if detection_time is not None else 0.0, 'rls_warmup_samples': warmup if detection_time is not None else None, 'rls_triggered': detection_time is not None, 'rls_forgetting_factor': forgetting, 'rls_sample_dt': sample_dt, 'control_signal': control_signal,
    }
    metrics['identification_quality'] = ('model-based estimate; not independently validated' if estimated_loe is not None else 'unavailable')
    metrics['detection_is_independent'] = False
    metrics['validation_only_ground_truth'] = True
    metrics['loe_absolute_error'] = abs(estimated_loe - loss) if estimated_loe is not None else None
    metrics['onset_absolute_error'] = abs(estimated_onset - fault_time) if estimated_onset is not None else None
    def json_safe(obj):
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return json_safe(obj.tolist())
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not math.isfinite(float(obj)):
            return None
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    (out / 'metrics.json').write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    def pair(label, key, normal, fault):
        return (f'{label}\n'
                f'  正常 / Normal: {fmt(normal[key])}\n'
                f'  故障 / Fault: {fmt(fault[key])}\n')
    report = (
        'SimDiag V3.7 ROBUSTNESS DEMO 多指标诊断报告 / Multi-Metric Diagnostic Report\n' + '=' * 65 + '\n'
        f'模型 / Model: {model}\n信号 / Signal: {signal}\n'
        f'故障类型 / Fault: 执行器效率损失 / Actuator LOE\n'
        f'故障开始 / Fault Onset: {fault_time:g} s\n'
        f'预设效率损失 / Configured LOE: {loss * 100:.1f}%\n'
        f'正常对象分母 / Plant Denominator: {normal_den}\n'
        f'健康模型预测RMSE / Healthy Model Prediction RMSE: {healthy_model_rmse:.6f}\n'
        f'模型一致性 / Model Consistency: {model_consistent} (tolerance {model_tolerance:.6f})\n\n'
        f'故障后分析窗口 / Post-fault Window: {start:g}–{common_end:g} s\n'
        f'响应偏差检测 / Response Deviation Detection: {detection}\n'
        f'故障后残差RMSE / Post-fault Residual RMSE: {residual_rmse:.4f}\n'
        f'故障后峰值残差 / Post-fault Peak Residual: {residual_peak:.4f}\n'
        f'故障后残差IAE / Post-fault Residual IAE: {residual_iae:.4f}\n'
        f'残差阈值 / Residual Threshold: ±{residual_threshold:.4f}\n'
        f'阈值方法 / Threshold Mode: {threshold_mode}\n'
        f'确认规则 / Confirmation Rule: {detection_rule}\n'
        f'检测时间 / Detection Time: {fmt(detection_time, 3)} s\n'
        f'检测延迟 / Detection Delay: {fmt(detection_delay, 3)} s\n\n'
        + pair('故障后RMSE / Post-fault RMSE', 'rmse', post_normal, post_fault)
        + pair('故障后IAE / Post-fault IAE', 'iae', post_normal, post_fault)
        + pair('故障后最大绝对误差 / Post-fault Maximum Absolute Error', 'max_abs_error', post_normal, post_fault)
        + pair('故障后窗口超调量 / Post-fault Window Overshoot', 'overshoot_absolute', post_normal, post_fault)
        + pair('故障后窗口调节时间(秒) / Post-fault Window Settling Time (s)', 'settling_time_from_window_start', post_normal, post_fault)
        + f'故障后RMSE变化 / Post-fault RMSE Change: {fmt(post_change, 2)}%\n'
        + f'性能评价 / Performance Assessment: {performance}\n\n'
        + f'因果递推RLS LOE估计 / LOE Estimate: {fmt(estimated_loe * 100 if estimated_loe is not None else None, 1)}%\n'
        + f'估计故障时刻 / Estimated Onset: {fmt(estimated_onset, 3)} s\n'
        + f'LOE绝对误差 / LOE Absolute Error: {fmt(abs(estimated_loe-loss)*100 if estimated_loe is not None else None, 1)} percentage points\n'
        + f'故障前误报 / Pre-onset False Alarm: {false_alarm}\n\n'
        + '说明 / Notes:\n'
        + '1. 检测触发后使用有限历史缓冲初始化候选起点RLS；每个时刻只使用当前及过去数据，不使用预设LOE或故障时间；仍属于离线回放。\n'
        + '   Causal RLS replay uses logged PID command and plant output; not live Simulink execution.\n'
        + '2. 检测残差来自故障运行输出减去由同次运行PID指令驱动的标称对象预测；并非两次仿真输出差。阈值未经噪声校准。\n'
        + '   Detection residual is fault-run output minus nominal prediction driven by fault-run PID command; not a live detector.\n'
        + '3. IAE是时间积分；RMSE是时间加权积分均方根，避免不同采样密度造成偏差。\n'
        + '   IAE is time-integrated; RMSE is time-weighted for irregular sample grids.\n'
        + '4. 超调量和调节时间针对故障后窗口计算，调节带为目标值绝对值的2%。\n'
        + '   Overshoot and settling use the post-fault window with a 2% reference band.\n'
        + '5. 未在窗口内进入并保持调节带时，调节时间显示N/A。\n'
        + '   Settling time is N/A when not settled before the window ends.\n'
        + '6. 估计准确性依赖对象分子、初始状态、执行器故障注入位置与记录的PID指令一致；须用盲测和扰动/噪声验证。\n' + '7. 本工具用于仿真对比，不是工业安全认证或现场故障确诊。\n'
        + '   Estimation assumes correct numerator, initial conditions, actuator location and logged PID command; independent blind validation is required.\n' + '   Simulation comparison only; not a certified safety or field diagnosis.\n'
    )
    report += '\n\nROBUSTNESS STUDY / 鲁棒性测试\n' + json.dumps(scenarios,ensure_ascii=False,indent=2) + '\nSensor noise is diagnostic-channel only; model mismatch uses separate Simulink runs. V3.7 uses paired healthy-reference calibration; NOT deployable standalone online detection.\n'
    (out / 'SimDiag_AI_Report.txt').write_text(report, encoding='utf-8')
    print(f'响应偏差检测 / Response deviation: {detection}', flush=True)
    print(f'检测时间 / Detection time: {fmt(detection_time, 3)} s; 检测延迟 / Delay: {fmt(detection_delay, 3)} s', flush=True)
    print(f'故障后RMSE变化 / Post-fault RMSE change: {fmt(post_change, 2)}%', flush=True)
    required = ('metrics.json', 'comparison.png', 'residual.png', 'loe_estimate.png', 'robustness.png', 'SimDiag_AI_Report.txt')
    missing = [f for f in required if not (out / f).is_file() or (out / f).stat().st_size == 0]
    if missing:
        raise RuntimeError('Incomplete robustness outputs: ' + ', '.join(missing))
    print('鲁棒性结果已保存 / Robustness outputs saved', flush=True)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
