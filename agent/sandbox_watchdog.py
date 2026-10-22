"""Trusted detached cleanup for one container; never reads generated code or data."""
import os
import re
import select
import subprocess
import sys
import time


def watch(executable, name, seconds, channel):
    if not os.path.isabs(executable) or not re.fullmatch(r'ets-research-[a-f0-9]{16}', name) or not 1 <= seconds <= 300:
        raise ValueError('Invalid sandbox cleanup arguments')
    select.select([channel], [], [], seconds)
    for attempt in range(3):
        try:
            subprocess.run([executable, 'rm', '-f', name], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if attempt < 2:
            time.sleep(1)


if __name__ == '__main__':
    watch(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.stdin)
