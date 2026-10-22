import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skills import experiment_research as er
from skills.llm_client import LLMError
from test_autonomous_research import FakeModel, reply


class ExperimentResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.session=er.ExperimentResearchSession(self.root,self.root/'S0000001',{})

    def test_multihop_research_then_next_round_reuses_with_correct_citations(self):
        model=FakeModel([reply([('search_web',{'query':'missing interaction XGBoost','engine':'bing'})]),
                         reply([('read_url',{'url':'https://example.com/paper'})]),
                         reply(text='可验证假设 [W1](https://example.com/paper)'),
                         reply(text='已有证据够用，复用 [R1](https://example.com/paper)')])
        def read(box,url):
            s=box.source({'url':url,'excerpt':'Missingness interactions','kind':'article_excerpt'})
            return {'source':s,'text':s['excerpt']}
        def search(box,query,engine):
            return {'results':[{'url':'https://example.com/paper'}]}
        updates=[]
        with patch.object(er.research_agent,'ToolModel',return_value=model), \
             patch.object(er.ExperimentTools,'search_web',search), \
             patch.object(er.ExperimentTools,'read_url',read):
            first=self.session.update(1,{'diagnostics':'missing data'},progress=updates.append)
            second=self.session.update(2,{'feedback':'no improvement'})
        self.assertEqual(first['decision'],'researched')
        self.assertEqual(len(first['trace']),2)
        self.assertTrue(updates)
        self.assertEqual(second['decision'],'reused')
        self.assertTrue(second['cached'])
        self.assertEqual(second['sources'][0]['source_id'],'R1')
        self.assertEqual(second['network_calls'],0)
        last=model.inputs[-1][0][-1]['content']
        self.assertIn('可验证假设 [R1]',last)
        self.assertNotIn('可验证假设 [W1]',last)
        saved=json.loads((self.session.directory/'web_research_02.json').read_text())
        self.assertEqual(saved['summary'],second['summary'])

    def test_model_failure_before_tools_is_not_classified_as_skipped(self):
        model=FakeModel([])
        with patch.object(er.research_agent,'ToolModel',return_value=model), \
             patch.object(model,'turn',side_effect=LLMError('unavailable')):
            report=self.session.update(1,{})
        self.assertEqual(report['decision'],'incomplete')
        self.assertEqual(report['status'],'unavailable')
        self.assertEqual(report['stop_reason'],'model_error')

    def test_bounded_history_keeps_newest_sources_and_marks_omitted_citations(self):
        previous={'sources':[{'source_id':f'W{i}','url':f'https://example.com/{i}'} for i in range(1,22)],
                  'summary':'新资料 [W1]，较早资料 [W21]'}
        box=er.ExperimentTools(self.root,'S1',0,previous)
        self.assertEqual(len(box.recalled),20)
        self.assertEqual(box.recalled[0]['url'],'https://example.com/1')
        self.assertEqual(box.previous_summary,'新资料 [R1]，较早资料 [来源未携带]')

    def test_skipping_is_a_model_decision_and_budget_forbids_tools(self):
        model=FakeModel([reply(text='无须补充资料'),reply(text='预算结束，证据不足')])
        with patch.object(er.research_agent,'ToolModel',return_value=model):
            skipped=self.session.update(1,{})
            budget=self.session.update(2,{},budget_seconds=0)
        self.assertEqual(skipped['decision'],'skipped')
        self.assertEqual(skipped['status'],'skipped')
        self.assertEqual(budget['stop_reason'],'budget')
        self.assertEqual(model.inputs[-1][1],'none')

    def test_interrupted_loop_checkpoints_completed_tools(self):
        model=FakeModel([reply([('recall_research',{})])])
        with patch.object(er.research_agent,'ToolModel',return_value=model):
            report=self.session.update(1,{})
        saved=json.loads((self.session.directory/'web_research_01.json').read_text())
        self.assertEqual(report['decision'],'failed')
        self.assertEqual(saved['trace'][0]['tool'],'recall_research')
        self.assertEqual(saved['trace'][0]['status'],'ok')
