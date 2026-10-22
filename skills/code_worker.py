"""Trusted Linux namespace launcher; user code runs only after chroot/capability drop."""
import ctypes
import errno
import json
import os
from pathlib import Path
import resource
import signal
import sys
import traceback


def checked(result):
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def launch(parent_pid, spec_path):
    libc = ctypes.CDLL(None, use_errno=True)
    checked(libc.prctl(1, signal.SIGKILL, 0, 0, 0))
    if os.getppid() != parent_pid:
        raise RuntimeError('Research owner has exited')
    uid, gid = os.getuid(), os.getgid()
    checked(libc.unshare(0x10000000))  # CLONE_NEWUSER
    Path('/proc/self/setgroups').write_text('deny')
    Path('/proc/self/uid_map').write_text(f'0 {uid} 1')
    Path('/proc/self/gid_map').write_text(f'0 {gid} 1')
    checked(libc.prctl(1, signal.SIGKILL, 0, 0, 0))
    if os.getppid() != parent_pid:
        raise RuntimeError('Research owner has exited')
    checked(libc.unshare(0x00020000 | 0x40000000 | 0x20000000 | 0x08000000 | 0x04000000))
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(writer)
        checked(libc.prctl(1, signal.SIGKILL, 0, 0, 0))
        os.set_blocking(reader, False)
        try:
            if os.read(reader, 1) == b'': os._exit(1)
        except BlockingIOError:
            pass
        os.close(reader)
        inside(spec_path)
        os._exit(1)
    os.close(reader)
    try:
        _, status = os.waitpid(pid, 0)
        sys.exit(os.waitstatus_to_exitcode(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status))
    finally:
        os.close(writer)


def isolate(spec):
    libc = ctypes.CDLL(None, use_errno=True)
    def mount(source, target, kind=None, flags=0, data=None):
        def arg(v): return os.fsencode(v) if v is not None else None
        checked(libc.mount(arg(source), arg(target), arg(kind), ctypes.c_ulong(flags), arg(data)))
    MS_BIND, MS_REC, MS_PRIVATE, MS_REMOUNT, MS_RDONLY = 4096, 16384, 1 << 18, 32, 1
    mount(None, '/', flags=MS_REC | MS_PRIVATE)
    jail = Path(spec['jail'])
    mount('tmpfs', jail, 'tmpfs', 2 | 4, 'size=128m,mode=755')
    def bind(source, target, readonly=True):
        source = Path(source)
        if not source.exists(): return
        target = jail / target.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir(): target.mkdir(exist_ok=True)
        else: target.touch()
        mount(source, target, flags=MS_BIND)
        mount(None, target, flags=MS_BIND | MS_REMOUNT | 2 | 4 | (MS_RDONLY if readonly else 0))
    # Only OS libraries and the selected interpreter, never home/project/data mounts.
    for folder in ['/usr', '/bin', '/lib', '/lib64', '/sbin']:
        bind(folder, folder)
    for folder in {sys.prefix, sys.base_prefix}:
        if folder not in ('/usr', '/'):
            bind(folder, folder)
    for path in ['/etc/ld.so.cache', '/etc/localtime', '/etc/fonts']:
        bind(path, path)
    bind(spec['work'], '/work', readonly=False)
    for device in ['null', 'zero', 'random', 'urandom']:
        # Device mount must allow access to these four specific devices.
        target = jail / 'dev' / device
        target.parent.mkdir(exist_ok=True); target.touch()
        mount('/dev/' + device, target, flags=MS_BIND)
    (jail / 'tmp').mkdir(mode=0o1777)
    (jail / 'proc').mkdir()
    # Keep /proc empty: this host forbids mounting proc in an unprivileged namespace.
    # No host process/filesystem handles are exposed to generated programs.
    os.chroot(jail); os.chdir('/work')
    # No ability to remount, escape chroot or change namespaces after this point.
    checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
    for cap in range(41):
        result = libc.prctl(24, cap, 0, 0, 0)  # bounding set
        if result < 0 and ctypes.get_errno() != errno.EINVAL: checked(result)
    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Caps(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
    checked(libc.capset(ctypes.byref(Header(0x20080522, 0)), ctypes.byref((Caps * 2)())))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (spec['memory_mb'] * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (spec['timeout'] + 1, spec['timeout'] + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(spec['env'])
    Path('/work/.home').mkdir(exist_ok=True)


def execute_notebook(path):
    """Execute Python cells in order using IPython; persist outputs even on cell failure."""
    import base64
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    nb = json.loads(path.read_text())
    shell = InteractiveShell.instance()
    shell.user_ns['__name__'] = '__main__'
    count, failed = 0, False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        count += 1
        cell['outputs'] = []; cell['execution_count'] = count
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        with capture_output() as captured:
            result = shell.run_cell(source, store_history=True)
            if 'matplotlib.pyplot' in sys.modules:
                import io
                from IPython.display import display, Image
                plt = sys.modules['matplotlib.pyplot']
                for number in plt.get_fignums():
                    buf = io.BytesIO(); plt.figure(number).savefig(buf, format='png')
                    display(Image(data=buf.getvalue()))
                plt.close('all')
        for name in ['stdout', 'stderr']:
            text = getattr(captured, name)
            if text:
                cell['outputs'].append({'output_type': 'stream', 'name': name, 'text': text[-32000:]})
                print(text[-32000:], end='', file=sys.stderr if name == 'stderr' else sys.stdout)
        for output in captured.outputs[:20]:
            data = {k: base64.b64encode(v).decode() if isinstance(v, bytes) else v
                    for k, v in output.data.items() if k in {'text/plain', 'text/html', 'image/png', 'image/jpeg', 'application/json'}}
            # Bound individual rich outputs to avoid oversized saved notebooks.
            if len(json.dumps(data)) <= 2_000_000:
                cell['outputs'].append({'output_type': 'display_data', 'data': data, 'metadata': output.metadata})
        error = result.error_before_exec or result.error_in_exec
        if error:
            cell['outputs'].append({'output_type': 'error', 'ename': type(error).__name__,
                                    'evalue': str(error), 'traceback': [str(error)]})
            failed = True
        path.with_name(path.stem + '.executed.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        if failed: break
    return 1 if failed else 0


def inside(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    isolate(spec)
    path = Path(spec['entry'])
    # Fresh isolated interpreter discards host modules, paths and launcher data.
    os.execv(sys.executable, [sys.executable, '-I', '-B', '/work/.runner.py', 'execute', str(path)])


def execute(entry):
    sys.path.insert(0, '/work')
    sys.argv = [entry]
    if entry.endswith('.ipynb'):
        return execute_notebook(Path(entry))
    import runpy
    runpy.run_path(entry, run_name='__main__')
    return 0


if __name__ == '__main__':
    action = sys.argv[1]
    try:
        if action == 'launch': launch(int(sys.argv[2]), sys.argv[3])
        elif action == 'inside': inside(sys.argv[2])
        elif action == 'execute': sys.exit(execute(sys.argv[2]))
        else: raise ValueError('Unknown runner action')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
