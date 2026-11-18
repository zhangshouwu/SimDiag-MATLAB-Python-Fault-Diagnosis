"""Compact bilingual SimDiag V3.7 ROBUSTNESS DEMO desktop GUI."""
import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

BASE = Path(__file__).resolve().parent
WORKER = BASE / 'sim_worker_v32.py'
REPORTS = BASE / 'reports_v37'
EXPECTED_VERSION = '3.7-robust-detection'
DEFAULT_MODEL = BASE / 'model' / 'SimDiag_PID_V14.slx'
if not DEFAULT_MODEL.is_file():
    DEFAULT_MODEL = BASE / 'SimDiag_PID_V14.slx'


class App:
    def __init__(self, root):
        self.root = root
        root.title('SimDiag V3.7 ROBUSTNESS DEMO - Intelligent Simulation & Fault Diagnosis')
        root.geometry('1150x740')
        root.minsize(830, 540)
        self.running = False
        self.photo = None
        self.image_source = None
        self.resize_job = None
        self.fields = {
            'model_path': tk.StringVar(value=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else ''),
            'block_path': tk.StringVar(value='Plant'),
            'profile_block': tk.StringVar(value='Fault_Profile'),
            'signal_name': tk.StringVar(value='plant_output'),
            'control_signal': tk.StringVar(value='pid_output'),
            'normal_den': tk.StringVar(value='[1 2 1]'),
            'step_time': tk.StringVar(value='1'),
            'reference': tk.StringVar(value='1'),
            'fault_time': tk.StringVar(value='3'),
            'fault_magnitude': tk.StringVar(value='0.3'),
            'stop_time': tk.StringVar(value='10'),
            'loe_grid_step': tk.StringVar(value='0.995'),
            'noise_sigma': tk.StringVar(value='0.02'),
            'mismatch_pct': tk.StringVar(value='20'),
            'noise_seed': tk.StringVar(value='2026'),
        }
        self.status = tk.StringVar(value='就绪 / Ready')
        self.metrics = [tk.StringVar(value='--') for _ in range(4)]
        self.residual_metric = tk.StringVar(value='--')
        self.detection_metric = tk.StringVar(value='检测时间 / Detection: --')
        self.estimate_metric = tk.StringVar(value='因果RLS回放 / Causal RLS Replay: --')
        self.build()
        # Do not automatically display old results on startup.

    def build(self):
        header = tk.Frame(self.root, bg='#17365D', height=67)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text='SimDiag V3.7 ROBUSTNESS DEMO', bg='#17365D', fg='white',
                 font=('Microsoft YaHei', 20, 'bold')).pack(pady=(2, 0))
        tk.Label(header, text='智能仿真与故障诊断系统 / Intelligent Simulation & Fault Diagnosis',
                 bg='#17365D', fg='white', font=('Microsoft YaHei', 9)).pack()

        config = ttk.LabelFrame(self.root, text='闭环执行器故障配置 / Closed-loop Actuator Fault Configuration', padding=6)
        config.pack(fill='x', padx=10, pady=4)
        ttk.Label(config, text='模型 / Model (.slx)').grid(row=0, column=0, sticky='w')
        ttk.Entry(config, textvariable=self.fields['model_path']).grid(row=0, column=1, columnspan=5,
                                                                        sticky='ew', padx=4)
        ttk.Button(config, text='导入 / Browse', command=self.browse).grid(row=0, column=6, padx=4)
        entries = [
            ('对象 / Plant', 'block_path'), ('故障模块 / Fault Block', 'profile_block'),
            ('信号 / Signal', 'signal_name'), ('控制信号 / PID Output', 'control_signal'), ('正常分母 / Den.', 'normal_den'),
            ('阶跃时间 / Step (s)', 'step_time'), ('目标值 / Reference', 'reference'),
            ('故障时间 / Fault (s)', 'fault_time'), ('损失比例 / LOE (0-1)', 'fault_magnitude'),
            ('停止时间 / Stop (s)', 'stop_time'),
            ('遗忘因子 / RLS λ', 'loe_grid_step'),
            ('噪声标准差 / Noise σ', 'noise_sigma'),
            ('参数偏差 / Mismatch %', 'mismatch_pct'),
            ('随机种子 / Seed', 'noise_seed'),
        ]
        for i, (label, name) in enumerate(entries):
            row, col = 1 + i // 3, (i % 3) * 2
            ttk.Label(config, text=label).grid(row=row, column=col, sticky='w', pady=3)
            ttk.Entry(config, textvariable=self.fields[name], width=14).grid(row=row, column=col + 1,
                                                                               sticky='ew', padx=4)
        for col in (1, 3, 5):
            config.columnconfigure(col, weight=1)

        actions = ttk.Frame(self.root)
        actions.pack(fill='x', padx=10, pady=2)
        self.run_button = ttk.Button(actions, text='运行仿真 / Run Simulation', command=self.run)
        self.run_button.pack(side='left', padx=3)
        ttk.Button(actions, text='打开报告 / Open Reports', command=self.open_reports).pack(side='left', padx=3)
        ttk.Button(actions, text='刷新 / Refresh', command=self.refresh).pack(side='left', padx=3)
        ttk.Label(actions, textvariable=self.status).pack(side='left', padx=12)

        metrics = ttk.LabelFrame(self.root, text='诊断指标 / Diagnostic Metrics', padding=5)
        metrics.pack(fill='x', padx=10, pady=3)
        labels = ['正常整体RMSE / Normal Overall RMSE', '故障整体RMSE / Fault Overall RMSE',
                  '故障后RMSE变化 / Post-fault Change', '性能评价 / Performance']
        for i, label in enumerate(labels):
            cell = ttk.Frame(metrics)
            cell.grid(row=0, column=i, sticky='ew', padx=3)
            metrics.columnconfigure(i, weight=1)
            ttk.Label(cell, text=label, anchor='center').pack()
            ttk.Label(cell, textvariable=self.metrics[i], anchor='center',
                      font=('Microsoft YaHei', 11, 'bold')).pack(pady=2)

        ttk.Label(self.root, textvariable=self.residual_metric, anchor='w').pack(fill='x', padx=15)
        ttk.Label(self.root, textvariable=self.detection_metric, anchor='w').pack(fill='x', padx=15)

        ttk.Label(self.root, textvariable=self.estimate_metric).pack(fill='x', padx=15)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=(3, 8))
        chart_tab = ttk.Frame(self.tabs)
        residual_tab = ttk.Frame(self.tabs)
        loe_tab = ttk.Frame(self.tabs)
        report_tab = ttk.Frame(self.tabs)
        log_tab = ttk.Frame(self.tabs)
        robustness_tab = ttk.Frame(self.tabs)
        self.tabs.add(chart_tab, text='响应曲线 / Response Chart')
        self.tabs.add(residual_tab, text='残差曲线 / Residual Chart')
        self.tabs.add(loe_tab, text='LOE因果RLS / Causal RLS')
        self.tabs.add(report_tab, text='诊断报告 / Diagnostic Report')
        self.tabs.add(log_tab, text='运行日志 / Execution Log')
        self.tabs.add(robustness_tab, text='抗干扰诊断 / Robustness')
        self.tabs.bind('<<NotebookTabChanged>>', lambda _event: self.root.after_idle(self.draw_image))
        self.chart = ttk.Label(chart_tab, text='暂无曲线 / No chart available', anchor='center')
        self.chart.pack(fill='both', expand=True)
        self.chart.bind('<Configure>', self.queue_resize)
        self.residual_chart = ttk.Label(residual_tab, text='暂无残差曲线 / No residual chart', anchor='center')
        self.residual_chart.pack(fill='both', expand=True)
        self.residual_chart.bind('<Configure>', self.queue_resize)
        self.loe_chart = ttk.Label(loe_tab, text='暂无LOE估计曲线 / No estimate yet', anchor='center')
        self.loe_chart.pack(fill='both', expand=True)
        self.loe_chart.bind('<Configure>', self.queue_resize)
        self.robustness_chart = ttk.Label(robustness_tab, text='暂无鲁棒性对比 / No robustness chart', anchor='center')
        self.robustness_chart.pack(fill='both', expand=True)
        self.robustness_chart.bind('<Configure>', self.queue_resize)
        self.robustness_source = None
        self.robustness_photo = None
        self.loe_source = None
        self.loe_photo = None
        self.residual_source = None
        self.residual_photo = None
        self.report = self.make_text(report_tab)
        self.log = self.make_text(log_tab)
        self.write_log('SimDiag V3.7 ROBUSTNESS DEMO 已启动 / Application started\n')

    @staticmethod
    def make_text(parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side='right', fill='y')
        text = tk.Text(frame, bg='#101820', fg='#00FF99', insertbackground='white',
                       font=('Microsoft YaHei', 10), wrap='word', yscrollcommand=scroll.set)
        text.pack(fill='both', expand=True)
        scroll.configure(command=text.yview)
        return text

    def write_log(self, text):
        self.log.insert('end', text)
        self.log.see('end')

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[('Simulink Models', '*.slx')])
        if path:
            self.fields['model_path'].set(path)

    def run(self):
        if self.running:
            return
        cfg = {key: var.get().strip() for key, var in self.fields.items()}
        model_path = Path(cfg['model_path'])
        if not model_path.is_file() or model_path.suffix.lower() != '.slx':
            messagebox.showerror('错误 / Error', '请选择有效的Simulink模型 / Select a valid .slx model')
            return
        if not WORKER.is_file():
            messagebox.showerror('错误 / Error', f'找不到工作程序 / Worker missing:\n{WORKER}')
            return
        try:
            values = {name: float(cfg[name]) for name in
                      ('step_time', 'reference', 'fault_time', 'fault_magnitude', 'stop_time', 'loe_grid_step', 'noise_sigma', 'mismatch_pct', 'noise_seed')}
            if not (0 <= values['fault_magnitude'] <= 1 and
                    0 <= values['fault_time'] < values['stop_time'] and
                    0 <= values['step_time'] < values['stop_time'] and
                     0.90 <= values['loe_grid_step'] <= 1.0 and 0 <= values['noise_sigma'] <= 0.5 and -50 <= values['mismatch_pct'] <= 50):
                raise ValueError('Invalid range')
        except ValueError:
            messagebox.showerror('错误 / Error', '请检查故障幅值、阶跃时间和停止时间 / Check numeric ranges')
            return
        REPORTS.mkdir(parents=True, exist_ok=True)
        cfg['output_dir'] = str(REPORTS)
        config_path = REPORTS / 'run_config.json'
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        self.running = True
        self.write_log(f'GUI: {Path(__file__).resolve()}\nWorker: {WORKER.resolve()}\nReports: {REPORTS.resolve()}\n')
        self.run_button.configure(state='disabled')
        self.status.set('正在仿真 / Simulation Running...')
        for var in self.metrics: var.set('--')
        self.estimate_metric.set('因果RLS回放 / Causal RLS Replay: --')
        self.residual_metric.set('--')
        self.detection_metric.set('检测时间 / Detection: --')
        self.clear_results()
        self.write_log('\n开始仿真 / Starting simulation...\n')
        self.tabs.select(4)
        threading.Thread(target=self.worker, args=(config_path,), daemon=True).start()

    def clear_results(self):
        # Clear images and report before a new run to prevent stale results.
        self.image_source = None
        self.residual_source = None
        self.loe_source = None
        self.robustness_source = None
        self.robustness_photo = None
        self.photo = None
        self.residual_photo = None
        self.loe_photo = None
        for widget, placeholder in (
            (self.chart, '暂无曲线 / No chart available'),
            (self.residual_chart, '暂无残差曲线 / No residual chart'),
            (self.loe_chart, '暂无LOE估计曲线 / No estimate yet'),
            (self.robustness_chart, '暂无鲁棒性对比 / No robustness chart'),
        ):
            widget.configure(image='', text=placeholder)
        self.report.delete('1.0', 'end')
        for filename in ('metrics.json', 'comparison.png', 'residual.png',
                         'loe_estimate.png', 'robustness.png', 'SimDiag_AI_Report.txt'):
            path = REPORTS / filename
            if path.exists():
                path.unlink()

    def worker(self, config_path):
        code = -1
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUNBUFFERED'] = '1'
            with subprocess.Popen([sys.executable, '-u', str(WORKER), '--config', str(config_path)],
                                  cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', env=env) as proc:
                for line in proc.stdout:
                    self.root.after(0, self.write_log, line)
                code = proc.wait()
        except Exception as exc:
            self.root.after(0, self.write_log, f'启动失败 / Launch failed: {exc}\n')
        self.root.after(0, self.finished, code)

    def finished(self, code):
        self.running = False
        self.run_button.configure(state='normal')
        if code == 0:
            try:
                self.validate_results()
            except Exception as exc:
                code = 2
                self.write_log(f'结果验证失败 / Output validation failed: {exc}\n')
        if code == 0:
            self.status.set('鲁棒性仿真完成 / Robustness Completed')
            self.refresh()
            self.tabs.select(5)
        else:
            self.status.set('仿真失败 / Simulation Failed')
            messagebox.showerror('仿真失败 / Simulation Failed',
                                 '请查看运行日志 / Please check the Execution Log')

    def validate_results(self):
        required = ['metrics.json', 'comparison.png', 'residual.png', 'loe_estimate.png', 'robustness.png', 'SimDiag_AI_Report.txt']
        missing = [name for name in required if not (REPORTS / name).is_file() or (REPORTS / name).stat().st_size == 0]
        if missing:
            raise RuntimeError('缺少输出文件 / Missing output files: ' + ', '.join(missing))
        data = json.loads((REPORTS / 'metrics.json').read_text(encoding='utf-8'))
        if data.get('version') != EXPECTED_VERSION:
            raise RuntimeError(f'版本不匹配 / Wrong worker version: {data.get("version")!r}, expected {EXPECTED_VERSION!r}')
        cases = data.get('robustness_scenarios')
        if not isinstance(cases, list) or len(cases) != 4:
            raise RuntimeError('鲁棒性测试结果不完整 / Four robustness scenarios required')
        with Image.open(REPORTS / 'robustness.png') as img:
            img.verify()

    def refresh(self):
        metrics_path = REPORTS / 'metrics.json'
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding='utf-8'))
                change = data.get('post_degradation')
                values = [f"{data['normal_rmse']:.4f}", f"{data['fault_rmse']:.4f}",
                          f'{change:+.2f}%' if change is not None else 'N/A', data['severity']]
                for var, value in zip(self.metrics, values):
                    var.set(value)
                estimate = data.get('estimated_loe')
                onset = data.get('estimated_onset')
                err = data.get('loe_absolute_error')
                self.estimate_metric.set((f'因果RLS回放 / Causal RLS Replay: {estimate*100:.2f}%    估计故障时刻 / Estimated onset: {onset:.2f} s' if onset is not None else f'因果RLS回放 / Causal RLS Replay: {estimate*100:.2f}%    Estimated onset: N/A') + (f'    验证误差 / Validation error: {err*100:.3f} pp' if err is not None else '') if estimate is not None else '因果RLS回放 / Causal RLS Replay: N/A')
                self.residual_metric.set('残差检测 / Residual Detection: ' + data.get('detection', '--') +
                                         '    峰值 / Peak: ' + format(data.get('residual_peak', 0), '.4f'))
                dt = data.get('detection_time')
                delay = data.get('detection_delay')
                threshold = data.get('residual_threshold')
                self.detection_metric.set(
                    f'检测时间 / Detection: {dt:.3f} s' if dt is not None else '检测时间 / Detection: N/A')
                self.detection_metric.set(self.detection_metric.get() +
                    (f'    延迟 / Delay: {delay:.3f} s' if delay is not None else '    延迟 / Delay: N/A') +
                    (f'    阈值 / Threshold: ±{threshold:.4f}' if threshold is not None else ''))
            except Exception as exc:
                self.write_log(f'指标加载失败 / Metrics error: {exc}\n')
        report_path = REPORTS / 'SimDiag_AI_Report.txt'
        if report_path.exists():
            self.report.delete('1.0', 'end')
            self.report.insert('end', report_path.read_text(encoding='utf-8', errors='replace'))
        residual_path = REPORTS / 'residual.png'
        if residual_path.exists():
            try:
                with Image.open(residual_path) as img:
                    self.residual_source = img.copy()
            except Exception as exc:
                self.write_log(f'残差图加载失败 / Residual image error: {exc}\n')
        loe_path = REPORTS / 'loe_estimate.png'
        if loe_path.exists():
            try:
                with Image.open(loe_path) as img:
                    self.loe_source = img.copy()
            except Exception as exc:
                self.write_log(f'LOE图加载失败 / LOE image error: {exc}\n')
        chart_path = REPORTS / 'comparison.png'
        if chart_path.exists():
            try:
                with Image.open(chart_path) as img:
                    self.image_source = img.copy()
            except Exception as exc:
                self.write_log(f'图片加载失败 / Image error: {exc}\n')

        robustness_path = REPORTS / 'robustness.png'
        if robustness_path.exists():
            try:
                with Image.open(robustness_path) as img:
                    self.robustness_source = img.copy()
            except Exception as exc:
                self.write_log(f'鲁棒性图加载失败 / Robustness image error: {exc}\n')
        self.draw_image()

    def queue_resize(self, _event):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self.draw_image)

    def draw_image(self):
        self.resize_job = None
        if self.image_source is not None:
            self._draw_chart(self.chart, self.image_source, 'photo')
        if self.loe_source is not None:
            self._draw_chart(self.loe_chart, self.loe_source, 'loe_photo')
        if self.robustness_source is not None:
            self._draw_chart(self.robustness_chart, self.robustness_source, 'robustness_photo')
        if self.residual_source is not None:
            self._draw_chart(self.residual_chart, self.residual_source, 'residual_photo')

    def _draw_chart(self, widget, source, attr):
        w = max(widget.winfo_width() - 12, 100)
        h = max(widget.winfo_height() - 12, 100)
        img = source.copy()
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        setattr(self, attr, photo)
        widget.configure(image=photo, text='')

    def open_reports(self):
        REPORTS.mkdir(parents=True, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(str(REPORTS))
        else:
            messagebox.showinfo('Reports', str(REPORTS))


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
