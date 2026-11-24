import json
import os
import stat
import sys
from .core import InvalidPlan, analyze

MAX_BYTES = 8 * 1024 * 1024


def pairs(items):
    result = {}
    for k, v in items:
        if k in result:
            raise InvalidPlan('repeated JSON object member')
        result[k] = v
    return result


def constant(_):
    raise InvalidPlan('non-JSON numeric constant')


def deliver(text):
    """One complete write plus flush, never treating a short write as delivery."""
    written = sys.stdout.write(text)
    if written != len(text):
        raise OSError('short output write')
    sys.stdout.flush()


def read_plan(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise InvalidPlan('plan input must be a regular file')
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InvalidPlan('plan exceeds 8 MiB')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _run():
    try:
        valid = len(sys.argv) == 2 or (len(sys.argv) == 4 and sys.argv[1] in ('--compare', '--receipts'))
        if not valid or sys.argv[1] in ('-h', '--help'):
            deliver('Usage: python -m batchscope PLAN.json\n       python -m batchscope --compare BEFORE.json AFTER.json\n       python -m batchscope --receipts PLAN.json RECEIPTS.json\n')
            return (0 if len(sys.argv) == 2 else 2), False
        if len(sys.argv) == 4 and sys.argv[1] == '--receipts':
            from .receipts import audit_receipts
            result = audit_receipts(read_plan(sys.argv[2]), read_plan(sys.argv[3]))
        elif len(sys.argv) == 4 and sys.argv[1] == '--compare':
            from .compare import compare_plans
            result = compare_plans(read_plan(sys.argv[2]), read_plan(sys.argv[3]))
        else:
            result = analyze(read_plan(sys.argv[1]))
        output = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n'
        deliver(output)
        return (0 if result['status'] in ('EXACTLY_ONCE', 'SAME_ACCOUNTING', 'MATCHED_RECEIPTS') else 1), False
    except (InvalidPlan, ValueError, OSError, RecursionError) as exc:
        try:
            sys.stderr.write('batchscope: invalid input or output failure: ' + str(exc) + '\n')
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        return 2, True


def main():
    """Return status without replacing or closing caller-owned streams."""
    return _run()[0]


def entrypoint():
    """Prevent a process-level retry of a failed buffered stdout at shutdown."""
    code, failed = _run()
    if failed:
        try:
            fd = sys.stdout.fileno()
            sink = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(sink, fd)
            finally:
                if sink != fd:
                    os.close(sink)
        except (OSError, ValueError, AttributeError):
            sys.stdout = open(os.devnull, 'w')
    return code


if __name__ == '__main__':
    raise SystemExit(entrypoint())
