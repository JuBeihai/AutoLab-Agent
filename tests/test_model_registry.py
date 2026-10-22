import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from skills.neural_model import fit, NeuralModel
from skills import model_registry, city_training

class ModelTests(unittest.TestCase):
    def test_preprocessing_reload_and_integrity(self):
        data = pd.DataFrame({'a': np.arange(60, dtype=float), 'b': np.nan, 'true_bin': np.tile(np.arange(6),10)})
        stop = data.copy(); stop['a'] = 10000.
        model, meta = fit(data, stop, ['a','b'], data.true_bin.to_numpy(), np.ones(60),
                          {'max_epochs':2, 'hidden_layers':[8]}, 6, 42, lambda _:None)
        self.assertEqual(model.mean[0], 29.5)
        self.assertEqual(model.median[1], 0)
        self.assertTrue(np.isfinite(model.predict(stop)).all())
        np.testing.assert_allclose(model.predict(stop).sum(axis=1),1,atol=1e-6)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/model_registry.filename('mlp')
            model.save(path,meta)
            np.testing.assert_array_equal(model.predict(stop),NeuralModel.load(path).predict(stop))
            path.write_bytes(path.read_bytes()+b'broken')
            with self.assertRaises(ValueError): NeuralModel.load(path)
    def test_registry_separation(self):
        for city in (None,*city_training.CITIES):
            self.assertNotEqual(city_training.latest_path('/tmp',city,'xgb'),city_training.latest_path('/tmp',city,'mlp'))
        with self.assertRaises(ValueError): model_registry.validate('unknown')
    def test_api_validation_and_forwarding(self):
        from unittest.mock import patch
        from webui.app import app
        with app.test_client() as client:
            self.assertEqual(client.post('/api/run',json={'model_family':'unknown'}).status_code,400)
            with patch('webui.runner.start_orchestrator',return_value='job_test') as start:
                response=client.post('/api/run',json={'model_family':'mlp','no_agent':True})
                self.assertEqual(response.status_code,200,response.json)
                self.assertEqual(start.call_args.kwargs['model_family'],'mlp')
            self.assertEqual(len(client.get('/api/models').json['catalog']),2)
    def test_mlp_config_persistence_validation_and_training(self):
        import json
        from unittest.mock import patch
        import webui.app as web
        from skills.neural_model import validate_config
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'train.json'
            target.write_text(json.dumps({'xgboost': {'max_depth':6}}))
            with patch.object(web,'CONFIG_DIR',Path(folder)), web.app.test_client() as client:
                params={'hidden_layers':[7,3], 'max_epochs':2, 'batch_size':12, 'learning_rate':.002}
                response=client.post('/api/config/train',json={'mlp_params':params})
                self.assertEqual(response.status_code,200,response.json)
                saved=json.loads(target.read_text())
                self.assertEqual(saved['xgboost'],{'max_depth':6})
                self.assertEqual(saved['mlp']['hidden_layers'],[7,3])
                before=target.read_bytes()
                for bad in [{'hidden_layers':[]},{'patience':0},{'max_epochs':1.5},{'batch_size':True},{'learning_rate':0},{'unknown':1}]:
                    self.assertEqual(client.post('/api/config/train',json={'mlp_params':bad}).status_code,400)
                    self.assertEqual(target.read_bytes(),before)
                data=pd.DataFrame({'a':np.arange(60,dtype=float),'true_bin':np.tile(np.arange(6),10)})
                model,meta=fit(data,data.iloc[:0],['a'],data.true_bin.to_numpy(),np.ones(60),saved['mlp'],6,42,lambda _:None)
                self.assertEqual([a.shape[1] for a in model.coefs],[7,3,6])
                self.assertEqual(meta['epochs_run'],2)
