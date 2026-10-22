import ast
import json
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from skills import mining_train, evaluation_views, isotonic_h3, model_registry, city_training
from skills.evaluation_policy import apply_training_population, reference_baseline, UNIFIED_CONTRACT, CONTRACT, SELECTION_KEY
ROOT=Path(__file__).resolve().parents[1]
class UnifiedTests(unittest.TestCase):
    def config(self):
        return apply_training_population(json.loads((ROOT/'configs/train.json').read_text()),'unified')
    def frame(self):
        return pd.DataFrame({'has_queue':np.arange(360)%2==0,'is_answer':np.where(np.arange(360)%11==0,0,1),
                             'true_bin':np.arange(360)%6,'signal':np.arange(360)%6})
    def test_notebook_split_parity_and_both_models(self):
        frame=self.frame();cfg=self.config()
        fit,stop=mining_train.notebook_split(frame,True,cfg)
        answered=frame[frame.is_answer==1];mask=np.random.RandomState(123).random_sample(len(answered))<.9
        self.assertEqual(fit.index.tolist(),answered.loc[mask].index.tolist())
        self.assertEqual(stop.index.tolist(),answered.loc[~mask].index.tolist())
        self.assertTrue((~fit.has_queue).any())
        for family in ['xgb','mlp']:
            cfg.update(model_family=family,mlp={'max_epochs':2,'hidden_layers':[8]})
            cfg['xgboost'].update(n_estimators=3,max_depth=2,early_stopping_rounds=2)
            with tempfile.TemporaryDirectory() as tmp:
                pairs=mining_train.train_pair(frame,['signal'],cfg,6,Path(tmp),{'experiment_id':'test','window':{}})
                self.assertIn('unified',pairs);self.assertNotIn('queue',pairs)
                pred=mining_train.predict(frame,pairs,['signal'])
                self.assertTrue(pred.predict_bin_44.between(0,5).all())
                loaded=model_registry.load(Path(tmp)/model_registry.filename(family,'unified'),family)
                np.testing.assert_array_equal(pred.predict_bin_44,mining_train.predict(frame,{'unified':loaded},['signal']).predict_bin_44)
        class ConstantModel:
            family='mlp'
            def predict(self, data): return np.tile([0,0,0,1,0,0],(len(data),1))
        self.assertTrue(mining_train.predict(frame,{'unified':ConstantModel()},['signal']).predict_bin_44.eq(3).all())
    def test_gate_and_inheritance_separation(self):
        contract={'schema':UNIFIED_CONTRACT,'source_sample_hash':'x','training_population':'unified','reference_baseline':reference_baseline('unified')}
        metric={city:{'total':100,'recall_ratio':.65} for city in ('ALL','100001','100002','100003')}
        base={'_evaluation_contract':contract,SELECTION_KEY:metric};trial=deepcopy(base)
        self.assertTrue(mining_train.gate(base,trial)['pass'])
        trial['_evaluation_contract']['schema']=CONTRACT
        self.assertFalse(mining_train.gate(base,trial)['pass'])
        self.assertNotEqual(city_training.latest_path(ROOT,None,'mlp','unified'),city_training.latest_path(ROOT,None,'mlp','queue_only'))
    def test_api_updates_population_and_rejects_invalid(self):
        from unittest.mock import patch
        import webui.app as web
        with tempfile.TemporaryDirectory() as tmp,patch.object(web,'CONFIG_DIR',Path(tmp)),web.app.test_client() as client:
            path=Path(tmp)/'train.json';path.write_text(json.dumps(self.config()))
            for population in ('queue_only','unified'):
                response=client.post('/api/config/train',json={'training_population':population})
                self.assertEqual(response.status_code,200)
                cfg=json.loads(path.read_text());self.assertEqual(cfg['non_queue_strategy'],'zero' if population=='queue_only' else 'unified')
            before=path.read_bytes()
            self.assertEqual(client.post('/api/config/train',json={'training_population':'bad','mlp_params':{'max_epochs':2}}).status_code,400)
            self.assertEqual(path.read_bytes(),before)
