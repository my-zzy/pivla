"""Terminal display shared by the trainer and its read-only log monitor."""
from collections import deque
import json
from pathlib import Path
import statistics

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


def duration(seconds):
    if seconds is None:
        return '—'
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:d}h {minutes:02d}m' if hours else f'{minutes:02d}m {seconds:02d}s'


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def tail_records(path, limit=50):
    """Read a bounded log tail, ignoring partial writes at both ends."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            offset = max(0, stream.tell() - 262144)
            stream.seek(offset)
            if offset:
                stream.readline()
            lines = stream.read().splitlines(keepends=True)
    except OSError:
        return []
    records = []
    for line in lines:
        if not line.endswith(b'\n'):
            continue
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeError):
            continue
    return records[-limit:]


def evaluation_text(metrics):
    return (f"position {metrics['position_rmse_m'] * 100:.2f} cm  ·  "
            f"yaw {metrics['yaw_rmse_rad']:.4f} rad  ·  "
            f"gripper {metrics['gripper_accuracy']:.1%}")


class TrainingDisplay:
    def __init__(self, total, batch, output, plain=False):
        self.total, self.batch, self.output = total, batch, Path(output)
        self.console = Console()
        self.interactive = self.console.is_terminal and not plain
        self.records = deque(maxlen=50)
        self.evaluation = {}
        self.phase_name = 'Starting'
        self.note = ''
        self.live = None

    def __enter__(self):
        if self.interactive:
            self.live = Live(console=self.console, get_renderable=self.render,
                             refresh_per_second=2)
            self.live.start()
        return self

    def __exit__(self, kind, value, tb):
        if kind:
            self.phase_name = 'Interrupted' if kind is KeyboardInterrupt else 'Failed'
            self.note = str(value)
        if self.live:
            self.live.stop()

    def phase(self, name, keep_note=False):
        self.phase_name = name
        if not keep_note:
            self.note = ''
        # Separate from status.json: this records a phase, not run success.
        path = self.output / 'progress.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'phase': name}) + '\n')
        temporary.replace(path)
        if not self.interactive:
            self.console.print(Text(name, style='cyan'))
            if keep_note and self.note:
                self.console.print(Text(self.note))

    def advance(self, done, total):
        self.note = f'{done:,}/{total:,} evaluation chunks'

    def update(self, record):
        self.records.append(record)
        if not self.interactive and (record['step'] == 1 or record['step'] % 10 == 0
                                     or record['step'] == self.total):
            speed, eta = self.timing()
            self.console.print(Text(
                f"Step {record['step']:,}/{self.total:,} | loss {record['loss']:.4f} "
                f"| lr {record['learning_rate']:.2e} | grad {record['grad_norm']:.2f} "
                f"| {speed:.3f} s/step | peak {record['peak_allocated_gib']:.2f} GiB "
                f"| optimizer ETA {duration(eta)}"))

    def evaluated(self, step, train, validation):
        self.evaluation = {'step': step, 'train': train, 'validation': validation}
        self.note = ''
        if not self.interactive:
            self.console.print(Text(f'Evaluation at step {step:,}'))
            self.console.print(Text('  Train: ' + evaluation_text(train)))
            if validation:
                self.console.print(Text('  Valid: ' + evaluation_text(validation)))

    def timing(self, records=None):
        records = list(self.records) if records is None else records
        if not records:
            return None, None
        recent = [r['step_s'] for r in records if r['step'] > 5] or [records[-1]['step_s']]
        speed = statistics.median(recent)
        return speed, max(0, self.total - records[-1]['step']) * speed

    def render(self):
        records = list(self.records)
        row = records[-1] if records else {}
        step = row.get('step', 0)
        speed, eta = self.timing(records)
        progress = f'{step:,}/{self.total:,} steps' if self.total else f'{step:,} steps'
        if self.total:
            progress += f'  ·  {step / self.total:.1%}'
        progress += f'  ·  batch {self.batch or "—"}'
        table = Table.grid(padding=(0, 2), expand=True)
        for _ in range(4):
            table.add_column()
        if row:
            mean = statistics.mean(r['loss'] for r in records)
            table.add_row('Loss', f"{row['loss']:.4f}", f'Mean ({len(records)})', f'{mean:.4f}')
            table.add_row('Learning rate', f"{row['learning_rate']:.2e}", 'Grad norm', f"{row['grad_norm']:.2f}")
            table.add_row('Step time (median)', f'{speed:.3f} s', 'Optimizer ETA', duration(eta))
            table.add_row('Training wall time', duration(row.get('elapsed_s')), 'Samples / s',
                          f"{row['batch_rows'] / speed:.1f}" if speed else '—')
            table.add_row('GPU peak allocated', f"{row['peak_allocated_gib']:.2f} GiB", 'Peak reserved',
                          f"{row['peak_reserved_gib']:.2f} GiB")
        else:
            table.add_row('Waiting for the first optimizer step', '', '', '')
        parts = [Text(self.phase_name, style='bold cyan'), Text(progress),
                 ProgressBar(total=max(1, self.total), completed=step), table]
        if self.evaluation:
            parts.append(Text(f"Latest evaluation · step {self.evaluation['step']:,}", style='bold'))
            for label, key in [('Train', 'train'), ('Valid', 'validation')]:
                if self.evaluation.get(key):
                    parts.append(Text(f'{label}: ' + evaluation_text(self.evaluation[key])))
        if self.note:
            parts.append(Text(self.note, style='yellow'))
        parts.extend([Text('ETA excludes future evaluation and checkpoint saving.', style='dim'),
                      Text(str(self.output), style='dim')])
        body = Table.grid(expand=True)
        for part in parts:
            body.add_row(part)
        return Panel(body, title='AirVLA · π0 training', border_style='cyan')
