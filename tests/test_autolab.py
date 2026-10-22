"""Generic tasks must train independently of waiting-time schemas and hold-out feedback."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from flask import Flask

from autolab.tasks import TaskSpec, load_task, local_path
from autolab.models import validate_candidate, utility, build, catalog, measure
from autolab.engine import run, validate_config, llm_candidate, research
from webui.lab_api import create_blueprint

ROOT = Path(__file__).resolve().parents[1]


class GenericTaskTests(unittest.TestCase):
    def test_both_task_types_have_disjoint_reproducible_partitions(self):
        for kind in ('classification', 'regression'):
            data = load_task(ROOT, TaskSpec.parse({'kind': kind}))
            sets = [set(x) for x in data['splits'].values()]
            self.assertEqual(len(set.union(*sets)), data['rows'])
            self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
            self.assertEqual(data['fingerprint'], load_task(ROOT, TaskSpec.parse({'kind':kind}))['fingerprint'])
            self.assertNotIn('true_bin', data['X'])
            self.assertNotIn('city_code', data['X'])

    def test_csv_categorical_and_time_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'local_data'; folder.mkdir()
            frame = pd.DataFrame({'timestamp': np.repeat(pd.date_range('2024-01-01', periods=30).astype(str), 4),
                                  'numeric': np.arange(120), 'segment': ['a', 'b', 'c', 'a']*30,
                                  'outcome': np.arange(120)%2})
            frame.to_csv(folder/'sample.csv', index=False)
            task = TaskSpec.parse({'dataset':'local_csv', 'file':'sample.csv', 'target':'outcome',
                                  'split':'time', 'time_column':'timestamp'})
            data = load_task(temp, task)
            self.assertEqual(set(data['feature_names'].values()), {'numeric','segment'})
            groups = [set(frame.iloc[rows].timestamp) for rows in data['splits'].values()]
            self.assertFalse(groups[0]&groups[1] or groups[1]&groups[2])
            candidate = validate_candidate({'model':'linear','params':{},'features':list(data['X']),
                                            'hypothesis':'test'}, 'classification', ['linear'], list(data['X']))
            model = build(candidate, 'classification', data['X'], 42)
            model.fit(data['X'].iloc[data['splits']['train']],data['y'][data['splits']['train']])
            self.assertEqual(len(model.predict(data['X'])),120)
            for name in ('../secret.csv', '/tmp/private.csv'):
                with self.assertRaises(ValueError): local_path(temp, name)
            (folder/'link.csv').symlink_to(folder/'sample.csv')
            with self.assertRaises(ValueError): local_path(temp,'link.csv')

    def test_registry_enforces_metric_and_parameter_contracts(self):
        with self.assertRaises(ValueError): TaskSpec.parse({'kind':'regression','metric':'accuracy'})
        with self.assertRaises(ValueError): validate_config({'rounds':True})
        with self.assertRaises(ValueError): validate_config({'models':['unknown']})
        for parameters in ({'n_jobs':64},{'max_depth':1000}):
            with self.assertRaises(ValueError):
                validate_candidate({'model':'xgb','params':parameters,'features':['f000'],
                                    'hypothesis':'test'},'classification',['xgb'],['f000'])
        self.assertGreater(utility({'rmse':2},'rmse'),utility({'rmse':3},'rmse'))
        self.assertGreater(utility({'accuracy':.8},'accuracy'),utility({'accuracy':.7},'accuracy'))


class GenericEngineTests(unittest.TestCase):
    def test_research_tool_boundary_and_public_only_request(self):
        task = TaskSpec.parse({'dataset': 'local_csv', 'file': 'private.csv', 'target': 'private_target'})
        def fake_research(root, config, messages, **kwargs):
            payload = json.dumps(messages)
            for private in ('private.csv', 'private_target', 'private_column', '0.9231'):
                self.assertNotIn(private, payload)
            toolbox = kwargs['toolbox']
            for tool in ('write_file', 'run_file', 'read_workspace_file', 'list_workspace'):
                with self.assertRaisesRegex(ValueError, '仅允许公开资料'):
                    toolbox.execute(tool, {})
            return SimpleNamespace(text='Public method summary', web_research={'sources': [], 'trace': []})
        with patch('skills.research_agent.run', side_effect=fake_research):
            report = research(ROOT, task, 1, {'validation': {'accuracy': .9231}},
                              config=validate_config({'planner': 'llm'}))
            self.assertEqual(report['mode'], 'llm_tool_research')
            self.assertEqual(report['summary'], 'Public method summary')

    def test_builtin_research_topics_are_supported_and_queries_are_public(self):
        from skills.web_research import TOPICS
        task = TaskSpec.parse({'kind': 'regression', 'dataset': 'local_csv', 'file': 'private.csv'})
        with patch('skills.web_research.collect', return_value={'sources': []}) as collect:
            for step, champion in [(1, None), (3, {'diagnostics': {}}),
                                   (1, {'diagnostics': {'train_validation_gap': .5}})]:
                research(ROOT, task, step, champion)
                request = collect.call_args.kwargs
                self.assertIn(request['topic'], TOPICS)
                self.assertIn('regression', request['query'])
                self.assertNotIn('private.csv', request['query'])

    def test_rounds_parent_evidence_and_no_test_feedback(self):
        snapshots = []
        def planner(step, state, config, features):
            snapshots.append(state)
            return {'model':'linear', 'params':{'C': [.1,10.][step]}, 'features':features,
                    'hypothesis':'A controlled parameter comparison'}
        with tempfile.TemporaryDirectory() as temp:
            state = run(ROOT, Path(temp)/'R0123456789ab', {'models':['linear'], 'rounds':2,
                        'trials_per_round':1,'web_research':True}, planner=planner,
                        researcher=lambda *args:{'sources':[{'title':'Fixture', 'url':'https://example.com/paper'}]})
            self.assertEqual(state['status'],'complete',state.get('error'))
            self.assertEqual(len(state['records']),3)
            self.assertEqual([row['round'] for row in state['records']],[0,1,2])
            self.assertTrue(all(x['test_metrics'] is None and not x['selection_locked'] for x in snapshots))
            self.assertTrue(all(row['parent_id'] in {r['id'] for r in state['records'][:i]}
                                for i,row in enumerate(state['records']) if i))
            self.assertEqual(len(state['research']),2)
            self.assertIn('accuracy',state['test_metrics'])
            self.assertTrue(state['records'][0]['diagnostics']['importance'])
            from autolab.replay import replay
            output = Path(temp)/'R0123456789ab'
            self.assertEqual(replay(output)['status'],'matched')
            (output/'splits.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'冻结输入'): replay(output)

    def test_every_registered_model_fits_classification_and_regression(self):
        for kind in ('classification','regression'):
            data = load_task(ROOT,TaskSpec.parse({'kind':kind}))
            X,y=data['X'],data['y'];train=data['splits']['train'];valid=data['splits']['validation']
            for name in catalog(kind):
                candidate = validate_candidate({'model':name,'params':{},'features':list(X),'hypothesis':'test'},kind,[name],list(X))
                model = build(candidate,kind,X,42)
                model.fit(X.iloc[train],y[train])
                metrics = measure(kind,y[valid],model.predict(X.iloc[valid]))
                self.assertTrue(np.isfinite(metrics['accuracy' if kind=='classification' else 'rmse']))

    def test_regression_and_graceful_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            state=run(ROOT,output,{'task':{'kind':'regression'},'models':['linear'],'rounds':0})
            self.assertEqual(state['status'],'complete',state.get('error'))
            self.assertIn('rmse',state['test_metrics'])
            from autolab.replay import replay
            self.assertEqual(replay(output)['status'],'matched')
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run';output.mkdir();(output/'cancel.flag').touch()
            state=run(ROOT,output,{'models':['linear'],'rounds':2})
            self.assertEqual(state['status'],'cancelled')
            self.assertIsNone(state['test_metrics'])

    def test_llm_receives_only_anonymous_features_and_aggregate_results(self):
        config=validate_config({'planner':'llm','task':{'dataset':'local_csv','file':'private.csv','target':'private_target'}})
        state={'records':[],'champion_id':'E0001','research':[], 'dataset':{'feature_names':{'f000':'private_column'}}}
        with patch('skills.llm_client.LLMClient') as client:
            client.return_value.chat.return_value=SimpleNamespace(text='{"model":"linear","params":{},"features":["f000"],"hypothesis":"test"}')
            llm_candidate(ROOT,state,config,['f000'])
            payload=client.return_value.chat.call_args.args[0][1]['content']
            for private in ('private.csv','private_target','private_column','test_metrics'):self.assertNotIn(private,payload)
            self.assertIn('f000',payload)


class GenericApiTests(unittest.TestCase):
    def test_validation_status_stop_and_invalid_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            app=Flask(__name__);app.register_blueprint(create_blueprint(temp))
            with app.test_client() as client:
                self.assertEqual(client.get('/api/lab/catalog').status_code,200)
                self.assertEqual(client.post('/api/lab/runs',json={'rounds':999}).status_code,400)
                process=Mock();process.poll.return_value=None
                with patch('webui.lab_api.subprocess.Popen',return_value=process):
                    response=client.post('/api/lab/runs',json={'models':['linear'],'rounds':0})
                    self.assertEqual(response.status_code,202,response.json)
                    identifier=response.json['id']
                    self.assertEqual(client.get('/api/lab/runs/'+identifier).json['status'],'queued')
                    self.assertEqual(client.post('/api/lab/runs',json={}).status_code,409)
                    self.assertEqual(client.post('/api/lab/runs/'+identifier+'/stop',json={}).status_code,200)
                    self.assertTrue((Path(temp)/'artifacts/autolab'/identifier/'cancel.flag').exists())
                self.assertEqual(client.get('/api/lab/runs/invalid').status_code,400)
