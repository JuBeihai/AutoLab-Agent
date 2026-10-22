from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from skills import city_training as cities, mining_train
from skills.evaluation_policy import apply_training_population, CONTRACT, SELECTION_KEY, reference_baseline

ROOT = Path(__file__).resolve().parents[1]


class CityTrainingTests(unittest.TestCase):
    def metrics(self, city, accuracy, mock=False):
        value = {'total': 100, 'recall_ratio': accuracy}
        return {SELECTION_KEY: {city: dict(value), 'ALL': dict(value)},
                '_evaluation_contract': {'schema': CONTRACT, 'source_sample_hash': city,
                    'training_city': city, 'mock': mock,
                    'reference_baseline': cities.scoped_reference(reference_baseline(), city)}}

    def test_city_thresholds_are_own_not_overall(self):
        reference = reference_baseline()
        for city in cities.CITIES:
            scoped = cities.scoped_reference(reference, city)
            self.assertAlmostEqual(scoped['metrics']['ALL']['recall_ratio'],
                                   reference['metrics'][city]['recall']/reference['metrics'][city]['total'])
        self.assertNotIn('training_city', reference)
        # A candidate can improve and still fall below the artificial 50% threshold.
        self.assertFalse(mining_train.gate(self.metrics('100002', .39), self.metrics('100002', .40))['pass'])
        # A candidate above the artificial threshold can be promoted.
        self.assertTrue(mining_train.gate(self.metrics('100003', .58), self.metrics('100003', .60))['pass'])

    def test_city_gates_are_independent_and_no_regression(self):
        outcomes = {city: mining_train.gate(self.metrics(city, base), self.metrics(city, trial))['pass']
                    for city, base, trial in [('100001', .64, .65), ('100002', .76, .75), ('100003', .58, .59)]}
        self.assertEqual(outcomes, {'100001': True, '100002': False, '100003': True})

    def test_contract_mismatch_empty_nan_and_unknown_city_rejected(self):
        base=self.metrics('100001', .64)
        for trial in [self.metrics('100003', .70), self.metrics('100001', float('nan'))]:
            self.assertFalse(mining_train.gate(base, trial)['pass'])
        trial=deepcopy(base);trial[SELECTION_KEY]['100001']['total']=0
        self.assertFalse(mining_train.gate(base, trial)['pass'])
        trial=deepcopy(base);trial['_evaluation_contract']['training_city']='x'
        self.assertFalse(mining_train.gate(base, trial)['pass'])

    def test_filter_preserves_duplicates_and_rejects_empty(self):
        frame=pd.DataFrame({'city_code':[100001,100002,100001,100003], 'order_id':['a','b','a','c']})
        original=frame.copy()
        selected=cities.select_city(frame,'100001')
        self.assertEqual(selected.order_id.tolist(), ['a','a'])
        pd.testing.assert_frame_equal(frame,original)
        with self.assertRaises(ValueError): cities.select_city(frame.iloc[:1], '100003')
        with self.assertRaises(ValueError): cities.assert_city(frame,'100001')

    def test_actual_models_saved_separately_and_reject_other_city_predictions(self):
        config=json.loads((ROOT/'configs/train.json').read_text())
        apply_training_population(config, 'queue_only')
        config.update(training_method='full_train',training_mode='by_city')
        config['xgboost'].update(n_estimators=2,n_jobs=1,max_depth=2)
        config['save']={}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for city in cities.CITIES:
                frame=pd.DataFrame({'city_code':[int(city)]*120,'has_queue':[True]*114+[False]*6,
                                    'is_answer':[1]*120,'true_bin':np.arange(120)%6,'feature':np.arange(120)%6})
                config['training_city']=city
                folder=Path(tmp)/city;folder.mkdir()
                models=mining_train.train_pair(frame,['feature'],config,6,folder,
                    {'experiment_id':'E0000001','window':{},'training_city':city})
                prediction=mining_train.predict(frame,models,['feature'])
                self.assertTrue((prediction.loc[~frame.has_queue,'predict_bin_44']==0).all())
                self.assertTrue((folder/'ets_pred_model_queue_44.json').is_file())
                self.assertFalse((folder/'ets_pred_model_no_queue_44.json').exists())
                with self.assertRaises(ValueError):
                    mining_train.predict(frame.assign(city_code=999999),models,['feature'])

    def test_failed_city_does_not_block_other_cities_or_overwrite_pointers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def run(argv):
                city=argv[argv.index('--city-code')+1];group=argv[-1];calls.append(city)
                sid='S000000'+str(len(calls))
                manifest={'session_id':sid,'training_city':city,'city_group_id':group,'status':'failed' if city=='100002' else 'complete'}
                cities.save(root/'experiments/feature_mining/current.json',manifest)
                if city=='100002':raise SystemExit(1)
                cities.save(cities.latest_path(root,city),manifest)
            with self.assertRaises(SystemExit),redirect_stdout(io.StringIO()):
                cities.run_group(root,['--training-mode','by_city'],run)
            self.assertEqual(calls,list(cities.CITIES))
            self.assertTrue(cities.latest_path(root,'100001').exists())
            self.assertTrue(cities.latest_path(root,'100003').exists())
            self.assertFalse(cities.latest_path(root,'100002').exists())
            self.assertFalse(cities.latest_path(root).exists())
            group=json.loads(next((root/'artifacts/city_training').glob('*.json')).read_text())
            self.assertEqual(group['status'],'partial_failure')

    def test_stop_does_not_start_remaining_cities(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def stop(argv):calls.append(argv);raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
                cities.run_group(Path(tmp),['--training-mode','by_city'],stop)
            self.assertEqual(len(calls),1)
            group=json.loads(next((Path(tmp)/'artifacts/city_training').glob('*.json')).read_text())
            self.assertEqual(group['status'],'cancelled')
            self.assertTrue(all(e['status']=='cancelled' for e in group['cities'].values()))

    def test_api_validates_and_forwards_mode(self):
        from webui import app as web
        client=web.app.test_client()
        for value in ('bad', [], None):
            self.assertEqual(client.post('/api/run',json={'mock':True,'training_mode':value}).status_code,400)
        with patch.object(web.runner,'start_orchestrator',return_value='job_test') as start:
            response=client.post('/api/run',json={'mock':True,'training_mode':'by_city'})
            self.assertEqual(response.status_code,200)
            self.assertEqual(start.call_args.kwargs['training_mode'],'by_city')
        self.assertEqual(client.get('/api/city-training/bad').status_code,400)


    def test_archiving_city_champion_only_rewinds_same_city_pointer(self):
        from webui.sessions import archive_session
        from webui import runner
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifacts=root/'artifacts';experiments=root/'experiments'
            for sid, city, date in [('S0000001','100001','2024-01-01'),
                                    ('S0000002','100001','2024-01-03'),
                                    ('S0000003','100003','2024-01-04')]:
                manifest={'session_id':sid,'status':'complete','mode':'feature_mining',
                          'training_city':city,'created_at':date,'champion_id':'E0000002','baseline_id':'E0000001'}
                cities.save(artifacts/'feature_mining'/sid/'session.json',manifest)
                cities.save(cities.latest_path(root,city),manifest)
            with patch.object(runner,'_jobs',{}):
                archive_session('S0000002',root,artifacts,experiments)
            self.assertEqual(json.loads(cities.latest_path(root,'100001').read_text())['session_id'],'S0000001')
            self.assertEqual(json.loads(cities.latest_path(root,'100003').read_text())['session_id'],'S0000003')
            self.assertFalse(cities.latest_path(root).exists())


if __name__ == '__main__':unittest.main()
