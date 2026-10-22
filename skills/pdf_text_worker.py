"""Run the installed PDF text extractor with process resource limits."""
import os
import resource
import sys


def main():
    executable, source, destination, pages = sys.argv[1:]
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execv(executable, [executable, '-f', '1', '-l', pages, '-enc', 'UTF-8', source, destination])


if __name__ == '__main__':
    main()
