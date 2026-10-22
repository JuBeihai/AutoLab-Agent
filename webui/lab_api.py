"""Local AutoLab job API; a separate process owns each training run."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
import psutil

from flask import Blueprint, jsonify, request
from autolab.engine import save, validate_config, now
from autolab.models import catalog
from autolab.tasks import TaskSpec, local_path


def create_blueprint(root):
    root = Path(root)
    blueprint = Blueprint('autolab', __name__, url_prefix='/api/lab')
    lock = threading.Lock()
    processes = {}

    def directory(identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r'R[0-9a-f]{12}', identifier):
            raise ValueError('非法实验会话编号')
        return root/'artifacts/autolab'/identifier

    def read(identifier):
        path = directory(identifier)/'run.json'
        if not path.is_file(): raise FileNotFoundError('实验会话不存在')
        state = json.loads(path.read_text())
        if state['status'] in ('running', 'queued'):
            process = processes.get(identifier)
            alive = process is not None and process.poll() is None
            if process is None and state.get('worker_pid'):
                try:
                    owner = psutil.Process(state['worker_pid'])
                    alive = owner.is_running() and owner.create_time() == state.get('worker_created_at')
                except psutil.Error:
                    pass
            if not alive:
                state.update(status='interrupted', stage='工作进程已退出；已有记录保留')
        state['stop_requested'] = (path.parent/'cancel.flag').exists()
        return state

    @blueprint.errorhandler(ValueError)
    def bad_request(error):
        return jsonify({'error': str(error)}), 400

    @blueprint.errorhandler(FileNotFoundError)
    def not_found(error):
        return jsonify({'error': str(error)}), 404

    @blueprint.get('/catalog')
    def capabilities():
        return jsonify({'name': 'AutoLab Agent', 'tasks': ['classification', 'regression'],
                        'models': catalog('classification'), 'default_planner': 'builtin'})

    @blueprint.get('/runs')
    def runs():
        entries = []
        base = root/'artifacts/autolab'
        for path in sorted(base.glob('R*/run.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:60]:
            state = read(path.parent.name)
            entries.append({k: state.get(k) for k in ('id', 'created_at', 'status', 'stage', 'config',
                                                     'champion_id', 'test_metrics', 'attempts', 'budget')})
        return jsonify({'runs': entries})

    @blueprint.get('/runs/<identifier>')
    def detail(identifier):
        response = jsonify(read(identifier)); response.headers['Cache-Control'] = 'no-store'
        return response

    @blueprint.post('/runs')
    def start():
        config = validate_config(request.get_json(silent=True))
        task = TaskSpec.parse(config['task'])
        if task.dataset == 'local_csv': local_path(root, task.file)
        if config['planner'] == 'llm':
            from skills.llm_profiles import resolve, key_env
            llm = resolve(json.loads((root/'configs/llm.json').read_text()), config['llm_profile'])
            if 'example.com' in llm['endpoint'] or llm['model'].startswith('YOUR_') or not os.environ.get(key_env(llm)):
                raise ValueError('请先在 configs/llm.json 和 .env 中配置自己的模型接口；也可选择内置策略体验')
        with lock:
            persisted = root/'artifacts/autolab'
            if any(p.poll() is None for p in processes.values()) or any(
                read(path.parent.name)['status'] in ('running', 'queued') for path in persisted.glob('R*/run.json')):
                return jsonify({'error': '已有实验运行中，请等待完成或停止后重试'}), 409
            identifier = 'R'+uuid.uuid4().hex[:12]
            output = directory(identifier); output.mkdir(parents=True)
            save(output/'config.json', config)
            save(output/'run.json', {'id': identifier, 'created_at': now(), 'status': 'queued',
                 'stage': '启动工作进程', 'config': config, 'records': [], 'research': [], 'events': [],
                 'champion_id': None, 'test_metrics': None, 'attempts': 0,
                 'budget': config['rounds']*config['trials_per_round'], 'round': 0})
            with (output/'worker.log').open('w') as log:
                try:
                    processes[identifier] = subprocess.Popen([sys.executable, '-m', 'autolab.cli', '--run-id', identifier],
                        cwd=root, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                except OSError:
                    save(output/'run.json', {'id': identifier, 'config': config, 'status': 'failed',
                                             'stage': '无法启动训练进程'})
                    raise
        return jsonify({'id': identifier}), 202

    @blueprint.post('/runs/<identifier>/stop')
    def stop(identifier):
        with lock:
            state = read(identifier)
            if state['status'] not in ('running', 'queued'):
                return jsonify({'status': state['status']})
            (directory(identifier)/'cancel.flag').write_text('stop requested\n')
        return jsonify({'status': 'stopping', 'message': '已请求停止；当前步骤结束后退出，保留已完成记录。'})

    return blueprint
