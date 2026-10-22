import copy
import json
from types import SimpleNamespace
from unittest.mock import patch
import test_mining_session as base
from agent import feature_mining
from skills import experiment_research as er
from skills import web_research as wr


def result(report=None, text='根据已有证据建议验证缺失值交叉特征。'):
    return SimpleNamespace(text=text, web_research=report, usage={})


class WebResearchSessionTests(base.MiningSessionTests):
    def test_autonomous_evidence_reaches_proposer_without_changing_samples(self):
        seen=[]
        report={'status':'ok', 'stop_reason':'complete',
                'sources':[{'source_id':'W1','url':'https://xgboost.readthedocs.io/', 'excerpt':'min_child_weight'}],
                'searches':[], 'errors':[], 'trace':[{'tool':'read_url','status':'ok'}]}
        def propose(profile,*args,**kwargs):
            seen.append(copy.deepcopy(profile['web_research']))
            return [self.proposals[0]]
        with patch.object(er.research_agent,'run',return_value=result(report)) as run, \
             patch.object(wr, 'collect', side_effect=AssertionError('legacy topic routing')), \
             patch.object(feature_mining,'propose',side_effect=propose):
            manifest,records,_=self.run_session(['--web-research','--rounds','1','--llm-profile','responses'])
        self.assertEqual(seen[0]['sources'][0]['source_id'],'W1')
        self.assertIn('建议验证',seen[0]['summary'])
        self.assertEqual(manifest['web_research_calls'],1)
        self.assertEqual(manifest['web_research_mode'],'autonomous_tools')
        self.assertEqual(run.call_args.args[1]['profile_id'],'responses')
        context=json.loads(run.call_args.args[2][-1]['content'])['context']
        self.assertIn('champion_diagnostics',context)
        self.assertTrue(context['allowed_changes']['training_config_fixed'])
        self.assertEqual(len({r['config_diff']['selection_sample_hash'] for r in records}),1)
        self.assertEqual(records[-1]['config_diff']['web_research_report'],'web_research_01.json')

    def test_disabled_baseline_and_predefined_proposals_do_not_research(self):
        fixture=self.root/'proposals.json'
        fixture.write_text(json.dumps({'proposals':self.proposals[:1]}))
        with patch.object(er.research_agent,'run') as run:
            self.run_session(['--web-research','--no-agent'])
            self.run_session(['--web-research','--proposal-file',str(fixture)])
            with patch.object(feature_mining,'propose',return_value=[self.proposals[0]]):
                self.run_session(['--rounds','1'])
        run.assert_not_called()

    def test_network_failure_does_not_abort_existing_research(self):
        with patch.object(er.research_agent,'run',side_effect=TimeoutError()), \
             patch.object(feature_mining,'propose',return_value=[self.proposals[0]]):
            manifest,records,_=self.run_session(['--web-research','--rounds','1'])
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual(manifest['web_research_status'],'unavailable')
        self.assertEqual(len(records),2)

    def test_every_round_can_decide_without_fixed_refresh_or_topic(self):
        contexts=[]
        def run(root,cfg,messages,**kwargs):
            contexts.append(json.loads(messages[-1]['content']))
            return result()
        with patch.object(er.research_agent,'run',side_effect=run), \
             patch.object(feature_mining,'propose',side_effect=[[p] for p in self.proposals[:2]]):
            manifest,records,_=self.run_session(['--web-research','--rounds','2'])
        self.assertEqual(len(contexts),2)
        self.assertEqual(contexts[0]['context']['recent_experiments'],[])
        self.assertTrue(contexts[1]['context']['recent_experiments'])
        self.assertEqual(manifest['web_research_calls'],2)
        self.assertEqual(manifest['web_research_status'],'skipped')
