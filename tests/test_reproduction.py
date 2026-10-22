import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from skills import reproduction as replay, mining_train
from skills.evaluation_views import build_views, selection_metrics, SELECTION_KEY

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def fixture(self, root, family='xgb'):
        cfg = json.loads((ROOT / 'configs/train.json').read_text())
        cfg.update(model_family=family, non_queue_strategy='unified', training_population='unified',
                   non_queue_after_ordering='model', training_method='notebook_random', model_seed=456,
                   mlp={'hidden_layers':[4], 'max_epochs':2, 'patience':2})
        cfg['xgboost'].update(n_estimators=5, max_depth=2, early_stopping_rounds=2)
        def frame(n, day):
            times = pd.date_range(day, periods=n, freq='min')
            return pd.DataFrame({'f': np.tile(np.arange(6), n // 6).astype(float),
                'order_id': [day + str(i) for i in range(n)], 'dt':day, 'city_code':100001,
                'has_queue':True, 'is_answer':1, 'true_bin':np.tile(np.arange(6), n // 6),
                'true_ets': np.tile(np.array([30,120,400,800,1400,2000]), n // 6),
                'create_time': times, 'match_time': times + pd.Timedelta(seconds=30),
                'cancel_time': pd.NaT, 'h3_res6':'a', 'h3_res7':'b', 'h3_res8':'c'})
        train, test = frame(600,'2024-08-01'), frame(60,'2024-08-02')
        out = root/'artifacts/feature_mining/S0000001/E0000001'
        out.mkdir(parents=True)
        replay.capture(root,out,train,test,['f'],cfg,protocol='feature_mining',window={})
        pairs=mining_train.train_pair(train,['f'],cfg,6,out,{'experiment_id':out.name,'window':{}})
        pred=mining_train.predict(test,pairs,['f']);metrics=mining_train.metrics(pred)
        views=build_views(pred,non_queue_after_ordering='model');metrics[SELECTION_KEY]=selection_metrics(views)
        record={'experiment_id':out.name,'config_diff':{'session_id':'S0000001','selection_metric':SELECTION_KEY},
                'metrics_by_city':metrics,'evaluation_views':views}
        replay.finalize(out,record)
        return out,record

    def test_portable_entry_retrains_both_families_without_live_repo(self):
        for family in ['xgb','mlp']:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                root=Path(folder);out,record=self.fixture(root/'original',family)
                migrated=root/'server with spaces'
                shutil.copytree(root/'original/artifacts',migrated/'artifacts')
                moved=migrated/out.relative_to(root/'original')
                env=os.environ.copy();env.pop('PYTHONPATH',None)
                result=subprocess.run([sys.executable,str(moved/'reproduce.py')],cwd='/tmp',env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                verified=replay.read(moved/'reproduction_retrain_result.json')
                self.assertEqual(verified['status'],'matched')
                self.assertEqual(verified['prediction_mismatches'],0)
                info=replay.detail(migrated,record)
                self.assertIn(str(migrated),info['entrypoint'])
                self.assertNotIn(str(root/'original'),info['command'])

    def test_modified_input_and_environment_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out,_=self.fixture(Path(folder))
            doc=replay.read(out/'reproduction.json')
            with patch.object(replay,'environment',return_value={**doc['environment'], 'packages':{}}), \
                 patch.object(mining_train,'train_pair') as train:
                self.assertEqual(replay.replay_main(out,[]),1)
                train.assert_not_called()
            path=out/doc['inputs']['train']['path']
            with path.open('ab') as f: f.write(b'tampered')
            with patch.object(mining_train,'train_pair') as train:
                self.assertEqual(replay.replay_main(out,[]),1)
                train.assert_not_called()
            self.assertIn('校验失败',replay.read(out/'reproduction_retrain_result.json')['error'])

    def test_metric_mismatch_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out,_=self.fixture(Path(folder))
            doc=replay.read(out/'reproduction.json')
            doc['expected_metrics'][SELECTION_KEY]['ALL']['total']+=1
            replay.write(out/'reproduction.json',doc)
            self.assertEqual(replay.replay_main(out,['--mode','evaluate']),1)
            result=replay.read(out/'reproduction_evaluate_result.json')
            self.assertEqual(result['status'],'mismatch')
            self.assertTrue(result['metric_mismatches'])

    def test_source_tampering_blocks_launcher(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out,_=self.fixture(Path(folder));doc=replay.read(out/'reproduction.json')
            source=out/doc['source']['path']/'skills/mining_train.py'
            source.write_text(source.read_text()+'\n# tampered\n')
            result=subprocess.run([sys.executable,str(out/'reproduce.py'),'--mode','evaluate'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('integrity check failed',result.stderr)


if __name__=='__main__':
    unittest.main()
