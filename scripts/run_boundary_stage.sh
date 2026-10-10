#!/usr/bin/env bash
set -eu

if [ ! -x .venv-cuda/bin/python ]; then
    printf '%s\n' 'Run from the WSL project root with .venv-cuda/bin/python available.' >&2
    exit 2
fi

exec .venv-cuda/bin/python - "$0" "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

SCRIPT = Path(sys.argv.pop(1)).resolve()
ROOT = Path.cwd().resolve()
if SCRIPT != ROOT / 'scripts/run_boundary_stage.sh':
    raise SystemExit('Run from the project root; the runner must be scripts/run_boundary_stage.sh.')
PYTHON = ROOT / '.venv-cuda/bin/python'
SOURCE = ROOT / 'reports/p4/training-source-20261010'
COARSE = ROOT / 'reports/p4/coarse-cache-cuda-20261010/result.json'
TRAIN = ROOT / 'reports/p4/training-cuda-20261010'
EVALUATE = ROOT / 'reports/p4/evaluation-cuda-20261010'
PLOTS = ROOT / 'reports/p4/plots-cuda-20261010'
ARCHIVE = ROOT / 'reports/p4/stage-runs-20261010'
AUDIO = ROOT / 'data/local/gtsinger-training-p4'
CONFIG = ROOT / 'configs/boundary-tcn.json'
STAGES = {
    'train': (TRAIN, 'result.json', 'boundary_training'),
    'evaluate': (EVALUATE, 'result.json', 'boundary_evaluation'),
    'plots': (PLOTS, 'index.json', 'boundary_diagnostics'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    value = json.loads(path.read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def write_json(path, value, *, replace=False):
    content = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if not replace:
        with path.open('x') as stream:
            stream.write(content)
        return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_birth(pid):
    try:
        stat = Path('/proc/{}/stat'.format(pid))
        if stat.exists():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            return None if fields[0] == 'Z' else 'proc:' + fields[19]
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='],
                                capture_output=True, text=True, check=False)
        return 'ps:' + result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, IndexError):
        return None


def alive(record):
    pid = record.get('pid')
    birth = record.get('birth')
    return type(pid) is int and pid > 0 and isinstance(birth, str) and process_birth(pid) == birth


def artifact_status(stage):
    directory, name, protocol = STAGES[stage]
    report = read_json(directory / name)
    if type(report.get('schema_version')) is not int or report['schema_version'] != 1 or report.get('protocol') != protocol:
        raise ValueError('Unexpected stage report schema/protocol: ' + stage)
    status = report.get('status')
    if status not in ('running', 'completed', 'failed'):
        raise ValueError('Missing or invalid stage report status: ' + stage)
    if stage == 'train' and status == 'completed' and not (directory / 'boundary.pt').is_file():
        raise ValueError('Completed training lacks boundary.pt')
    return status


def inspect(stage):
    archive = ARCHIVE / stage
    output = STAGES[stage][0]
    result = {'stage': stage, 'archive': str(archive), 'output': str(output), 'success': False}
    if not archive.exists():
        if output.exists():
            result.update(status='external_unverified', reason='Output exists without runner exit/status evidence; no overwrite.')
            try:
                result['artifact_status'] = artifact_status(stage)
            except (OSError, ValueError) as error:
                result['artifact_error'] = str(error)
        else:
            result['status'] = 'not_started'
        return result
    try:
        launch = read_json(archive / 'launch.json')
        state = read_json(archive / 'status.json')
        if (launch.get('stage') != stage or state.get('stage') != stage
                or launch.get('token') != state.get('token')):
            raise ValueError('Launch/status identity mismatch')
        result['log'] = str(archive / 'stage.log')
        exit_path = archive / 'exit.json'
        if exit_path.exists():
            exit_record = read_json(exit_path)
            code = exit_record.get('exit_code')
            if (type(code) is not int or code < 0 or exit_record.get('token') != launch['token']
                    or exit_record.get('stage') != stage):
                raise ValueError('Invalid exit record')
            result['exit_code'] = code
            if code == 0 and state.get('status') == 'completed' and artifact_status(stage) == 'completed':
                result.update(status='completed', success=True)
            else:
                result.update(status='failed', reason='Exit/status/artifact do not jointly establish completion.')
            return result
        pid_path = archive / 'pid.json'
        if pid_path.exists():
            pid = read_json(pid_path)
            if pid.get('token') != launch['token'] or pid.get('stage') != stage:
                raise ValueError('PID identity mismatch')
            result['pid'] = pid.get('pid')
            if state.get('status') == 'running' and alive(pid):
                result['status'] = 'running'
                return result
        elif state.get('status') == 'starting':
            spawn_path = archive / 'spawn.json'
            spawn = read_json(spawn_path) if spawn_path.exists() else {}
            if spawn and (spawn.get('stage') != stage or spawn.get('token') != launch['token']):
                raise ValueError('Spawn identity mismatch')
            if alive(spawn) or alive(launch.get('launcher', {})):
                result.update(status='starting', pid=spawn.get('pid'))
                return result
        result.update(status='incomplete', reason='No live worker with matching process birth, or missing exit/status evidence.')
    except (OSError, ValueError, KeyError) as error:
        result.update(status='incomplete', reason=str(error))
    return result


def command(stage):
    common = ['--segments', str(SOURCE / 'segments.jsonl'), '--splits', str(SOURCE / 'splits.json'),
              '--audio-root', str(AUDIO), '--coarse-report', str(COARSE)]
    if stage == 'train':
        return [str(PYTHON), str(ROOT / 'scripts/train_boundary.py')] + common + [
            '--config', str(CONFIG), '--out-dir', str(TRAIN)]
    if stage == 'evaluate':
        return [str(PYTHON), str(ROOT / 'scripts/evaluate_boundary.py')] + common + [
            '--checkpoint', str(TRAIN / 'boundary.pt'), '--device', 'cuda', '--batch-size', '32',
            '--out-dir', str(EVALUATE)]
    return [str(PYTHON), str(ROOT / 'scripts/plot_boundary_results.py'),
            '--training-report', str(TRAIN / 'result.json'), '--evaluation-report', str(EVALUATE / 'result.json'),
            '--out-dir', str(PLOTS), '--failure-cases', '--segments', str(SOURCE / 'segments.jsonl'),
            '--splits', str(SOURCE / 'splits.json'), '--audio-root', str(AUDIO)]


def require_inputs(stage):
    for prerequisite in (() if stage == 'train' else ('train',) if stage == 'evaluate' else ('train', 'evaluate')):
        result = inspect(prerequisite)
        if not result['success']:
            raise ValueError('Prerequisite is not verified completed: ' + json.dumps(result))
    for path in (SOURCE / 'segments.jsonl', SOURCE / 'splits.json', COARSE):
        if not path.is_file():
            raise ValueError('Missing fixed input: ' + str(path))
    if not AUDIO.is_dir():
        raise ValueError('Missing fixed audio root: ' + str(AUDIO))
    if stage == 'train' and not CONFIG.is_file():
        raise ValueError('Missing fixed config: ' + str(CONFIG))
    if not Path(command(stage)[1]).is_file():
        raise ValueError('Missing stage entry point')


def worker(stage, token):
    archive = ARCHIVE / stage
    launch = read_json(archive / 'launch.json')
    if launch.get('stage') != stage or launch.get('token') != token or launch.get('command') != command(stage):
        raise ValueError('Worker launch identity/command mismatch')
    if STAGES[stage][0].exists():
        raise ValueError('Output appeared before worker start; refusing overwrite')
    pid = {'stage': stage, 'token': token, 'pid': os.getpid(), 'birth': process_birth(os.getpid())}
    write_json(archive / 'pid.json', pid)
    write_json(archive / 'status.json', {'stage': stage, 'token': token, 'status': 'running', 'started_at': now()}, replace=True)
    code, error, child = 1, None, None
    try:
        require_inputs(stage)
        child = subprocess.Popen(command(stage), cwd=ROOT, stdin=subprocess.DEVNULL)

        def stop(signum, frame):
            if child.poll() is None:
                child.send_signal(signum)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        code = child.wait()
        if code < 0:
            code = 128 - code
        if code == 0 and artifact_status(stage) != 'completed':
            raise ValueError('Command exited zero without a completed stage report')
    except Exception as exception:
        code, error = 1, type(exception).__name__ + ': ' + str(exception)
        print(error, file=sys.stderr, flush=True)
    final = {'stage': stage, 'token': token, 'status': 'completed' if code == 0 else 'failed', 'finished_at': now()}
    if error is not None:
        final['error'] = error
    write_json(archive / 'exit.json', dict(final, exit_code=code))
    write_json(archive / 'status.json', final, replace=True)
    return code


def start(stage, foreground):
    existing = inspect(stage)
    if existing['status'] != 'not_started':
        print(json.dumps(existing, indent=2))
        return 0 if existing['status'] in ('starting', 'running', 'completed') else 1
    require_inputs(stage)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE / stage
    try:
        archive.mkdir()
    except FileExistsError:
        result = inspect(stage)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] in ('starting', 'running', 'completed') else 1
    token = uuid.uuid4().hex
    launch = {'schema_version': 1, 'stage': stage, 'token': token, 'created_at': now(), 'command': command(stage),
              'runner_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
              'launcher': {'pid': os.getpid(), 'birth': process_birth(os.getpid())}}
    write_json(archive / 'launch.json', launch)
    write_json(archive / 'status.json', {'stage': stage, 'token': token, 'status': 'starting'})
    with (archive / 'stage.log').open('x') as log:
        try:
            # nohup plus a new session leaves the worker independent of the SSH terminal.
            child = subprocess.Popen(['nohup', 'bash', str(SCRIPT), '_worker', stage, token],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            write_json(archive / 'spawn.json', {'stage': stage, 'token': token, 'pid': child.pid,
                                              'birth': process_birth(child.pid)})
        except OSError as error:
            final = {'stage': stage, 'token': token, 'status': 'failed', 'finished_at': now(), 'error': str(error)}
            write_json(archive / 'exit.json', dict(final, exit_code=1))
            write_json(archive / 'status.json', final, replace=True)
            raise
    if foreground:
        child.wait()
        result = inspect(stage)
        print(json.dumps(result, indent=2))
        return 0 if result['success'] else 1
    print(json.dumps({'stage': stage, 'status': 'started', 'success': False, 'launcher_worker_pid': child.pid,
                      'archive': str(archive), 'log': str(archive / 'stage.log'),
                      'check': 'bash scripts/run_boundary_stage.sh status'}, indent=2))
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '_worker' and sys.argv[2] in STAGES:
        return worker(sys.argv[2], sys.argv[3])
    parser = argparse.ArgumentParser(description='Run/recover fixed P4 stages; existing archives are never restarted.')
    parser.add_argument('stage', choices=['train', 'evaluate', 'plots', 'status'])
    parser.add_argument('--foreground', action='store_true', help='Wait and return verified stage success/failure.')
    args = parser.parse_args()
    if args.stage == 'status':
        if args.foreground:
            parser.error('--foreground is only valid for train/evaluate/plots')
        print(json.dumps({stage: inspect(stage) for stage in STAGES}, indent=2))
        return 0
    return start(args.stage, args.foreground)


try:
    raise SystemExit(main())
except (OSError, ValueError, KeyError) as error:
    print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
    raise SystemExit(1)
PY
