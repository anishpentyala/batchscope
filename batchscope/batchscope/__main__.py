import json
import io
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


def main():
    try:
        if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
            sys.stdout.write('Usage: python -m batchscope PLAN.json\n')
            sys.stdout.flush()
            return 0 if len(sys.argv) == 2 else 2
        path = sys.argv[1]
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise InvalidPlan('plan input must be a regular file')
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise InvalidPlan('plan exceeds 8 MiB')
        plan = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        result = analyze(plan)
        output = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n'
        sys.stdout.write(output)
        sys.stdout.flush()
        return 0 if result['status'] == 'EXACTLY_ONCE' else 1
    except (InvalidPlan, ValueError, OSError, RecursionError) as exc:
        if isinstance(exc, OSError):
            # The console-script wrapper must not retry a failed stream at exit.
            sys.stdout = io.StringIO()
        try:
            sys.stderr.write('batchscope: invalid input or output failure: ' + str(exc) + '\n')
            sys.stderr.flush()
        except OSError:
            pass
        return 2


if __name__ == '__main__':
    status = main()
    if status == 2:
        # Avoid a second stdout flush changing output-error status at interpreter exit.
        try:
            sys.stdout.close()
        except OSError:
            pass
    raise SystemExit(status)
