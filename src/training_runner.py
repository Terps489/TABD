"""Запуск train/predict/evaluate как фонового процесса для дашборда.

Дашборд работает в одном Python-процессе и не должен блокироваться обучением,
поэтому каждое задание стартует через `subprocess.Popen(run.py --mode ...)`.
PID и путь к лог-файлу сохраняются в `logs/runner/` — это позволяет UI:
- знать, есть ли активная задача (`is_running`)
- показывать "хвост" лога в реальном времени (`tail_log`)
- защитить от параллельных запусков (если процесс жив — не пускаем новый)

Один runner-state на проект (один процесс одновременно).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import LOGS_DIR, PROJECT_DIR


RUNNER_DIR = LOGS_DIR / "runner"
STATE_FILE = RUNNER_DIR / "state.json"

VALID_MODES = ("train", "predict", "evaluate")


@dataclass
class RunnerState:
    pid: int
    mode: str
    quick: bool
    started_at: float
    log_path: str
    finished_at: float | None = None
    exit_code: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _ensure_dirs() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)


def _read_state() -> RunnerState | None:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return RunnerState(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def _write_state(state: RunnerState) -> None:
    _ensure_dirs()
    STATE_FILE.write_text(state.to_json(), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    """Проверка, что процесс жив. Кроссплатформенно."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, check=False,
            )
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _refresh_state(state: RunnerState) -> RunnerState:
    """Если pid умер, проставляем finished_at / exit_code (грубо, без waitpid)."""
    if state.finished_at is None and not _pid_alive(state.pid):
        state.finished_at = time.time()
        if state.exit_code is None:
            state.exit_code = -1  # неизвестен — процесс пропал между опросами
        _write_state(state)
    return state


def is_running() -> bool:
    s = _read_state()
    if s is None:
        return False
    s = _refresh_state(s)
    return s.finished_at is None


def get_status() -> dict:
    s = _read_state()
    if s is None:
        return {"running": False, "state": None}
    s = _refresh_state(s)
    return {
        "running": s.finished_at is None,
        "state": asdict(s),
    }


def tail_log(n_lines: int = 100) -> str:
    s = _read_state()
    if s is None:
        return ""
    log_path = Path(s.log_path)
    if not log_path.exists():
        return ""
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 64 * 1024)
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def start(mode: str, quick: bool = False) -> RunnerState:
    """Запустить run.py в подпроцессе. Бросает RuntimeError, если уже идёт задача."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode должен быть из {VALID_MODES}, получено {mode!r}")

    if is_running():
        raise RuntimeError(
            "Уже выполняется другое задание. Дождитесь завершения или остановите его."
        )

    _ensure_dirs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = RUNNER_DIR / f"{mode}_{timestamp}.log"

    cmd = [sys.executable, str(PROJECT_DIR / "run.py"), "--mode", mode]
    if quick and mode in ("train",):
        cmd.append("--quick")

    env = os.environ.copy()
    # Чтобы не падало на cp1251 в stdout (Lightning любит писать emoji).
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(f"$ {' '.join(cmd)}\n")
    log_file.flush()

    popen_kwargs = dict(
        stdout=log_file, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR), env=env,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, **popen_kwargs)
    state = RunnerState(
        pid=proc.pid, mode=mode, quick=quick,
        started_at=time.time(), log_path=str(log_path),
    )
    _write_state(state)
    return state


def stop() -> bool:
    """Аккуратная остановка текущего процесса. True если что-то остановили."""
    s = _read_state()
    if s is None or s.finished_at is not None:
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(s.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            os.kill(s.pid, signal.SIGTERM)
        s.finished_at = time.time()
        s.exit_code = -2  # остановлено пользователем
        _write_state(s)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def clear() -> None:
    """Удалить state-файл — следующий start стартует с чистого листа."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
