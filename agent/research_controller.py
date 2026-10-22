"""Bounded sequential research planning; metrics remain the promotion authority."""
from __future__ import annotations

import time


DIRECTIONS = ('supply_demand', 'temporal_change', 'history_coverage', 'spatial_context', 'feature_interaction')
RESEARCH_DIRECTIONS = ('feature_selection', 'missing_encoding', 'training_strategy', 'database_inspection')
ACTIONS = {'explore', 'refine', 'ablate', 'combine', 'repair', 'stop'}


class ResearchController:
    def __init__(self, max_steps, timeout_minutes=120, patience=9, clock=time.monotonic, research_skills=False):
        self.max_steps = max_steps
        self.timeout_seconds = timeout_minutes * 60
        self.patience = patience
        self.clock = clock
        self.started = clock()
        self.steps = []
        self.no_improvement = 0
        self.planning_failures = 0
        self.stop_reason = None
        self.research_skills = research_skills
        self.small_gain_steps = 0

    def stopped(self):
        if self.stop_reason:
            return True
        if len(self.steps) >= self.max_steps:
            self.stop_reason = '已达到提案／实验尝试预算'
        elif self.clock() - self.started >= self.timeout_seconds:
            self.stop_reason = '搜索时间预算已用尽；不再安排新实验'
        elif self.no_improvement >= self.patience:
            self.stop_reason = f'连续 {self.patience} 次尝试未提升全局冠军，停止搜索并保留结果'
        elif self.planning_failures >= 3:
            self.stop_reason = '连续三次规划失败，请检查模型服务或输出格式'
        return self.stop_reason is not None

    def snapshot(self):
        previous = self.steps[-1] if self.steps else {}
        recent_repairs = 0
        for step in reversed(self.steps):
            if step.get('plan', {}).get('action') != 'repair':
                break
            recent_repairs += 1
        repair_allowed = previous.get('status') == 'execution_failed' and recent_repairs < 2
        direction = DIRECTIONS[(self.no_improvement // 2) % len(DIRECTIONS)]
        if previous.get('improved'):
            direction = previous.get('plan', {}).get('direction', direction)
        if repair_allowed:
            direction = previous.get('plan', {}).get('direction', direction)
        diversify = self.research_skills and self.small_gain_steps >= 3 and not repair_allowed
        if diversify:
            tried = {step.get('plan', {}).get('direction') for step in self.steps[-3:]}
            direction = next((name for name in RESEARCH_DIRECTIONS if name not in tried), 'feature_selection')
        return {'mode': 'continuous', 'max_steps': self.max_steps, 'attempted_steps': len(self.steps),
                'remaining_steps': max(0, self.max_steps - len(self.steps)),
                'remaining_seconds': max(0, int(self.timeout_seconds - (self.clock() - self.started))),
                'no_improvement_steps': self.no_improvement, 'patience': self.patience,
                'repair_allowed': repair_allowed,
                'recommended_action': 'repair' if repair_allowed else 'refine' if previous.get('improved') else 'explore',
                'recommended_direction': direction, 'last_steps': self.steps[-5:],
                'small_gain_steps': self.small_gain_steps,
                'require_alternative_to_add_only': diversify,
                'stop_reason': self.stop_reason}

    def validate(self, result):
        if not isinstance(result, dict) or set(result) != {'plan', 'proposals'}:
            raise ValueError('持续探索必须返回 plan 和 proposals')
        plan = result['plan']
        if not isinstance(plan, dict) or set(plan) != {'action', 'direction', 'reason', 'next_if_failed'}:
            raise ValueError('plan 必须包含 action/direction/reason/next_if_failed')
        allowed = DIRECTIONS + RESEARCH_DIRECTIONS if self.research_skills else DIRECTIONS
        if plan['action'] not in ACTIONS or plan['direction'] not in allowed:
            raise ValueError('未知研究动作或方向')
        for key in ('reason', 'next_if_failed'):
            if not isinstance(plan[key], str) or not 1 <= len(plan[key].strip()) <= 1500:
                raise ValueError('研究理由和后备计划不能为空或过长')
        if plan['action'] == 'repair' and not self.snapshot()['repair_allowed']:
            raise ValueError('仅允许修复上一项执行失败；连续修复最多两次')
        proposals = result['proposals']
        if not isinstance(proposals, list) or len(proposals) != (0 if plan['action'] == 'stop' else 1):
            raise ValueError('每次只安排一个实验；stop 时 proposals 必须为空')
        if proposals and self.snapshot()['require_alternative_to_add_only']:
            proposal = proposals[0]
            if not isinstance(proposal, dict):
                raise ValueError('候选必须是对象')
            alternative = any(proposal.get(key) for key in ('remove_features', 'training', 'inspect', 'requests', 'extensions'))
            alternative |= any(isinstance(recipe, dict) and recipe.get('kind') in {
                'missing_indicator', 'impute', 'one_hot', 'target_encoding'} for recipe in proposal.get('feature_recipes', []))
            if not alternative:
                raise ValueError('连续三次无收益或提升不足0.1pp，下一步须验证删减、编码、训练策略或只读探查，不能继续仅加普通配方')
        return plan, proposals

    def record(self, plan, status, improved=False, experiment_id=None, detail=None, delta_pp=None):
        self.steps.append({'step': len(self.steps) + 1, 'plan': plan, 'status': status,
                           'improved': improved, 'experiment_id': experiment_id, 'detail': detail, 'delta_pp': delta_pp})
        if status not in {'inspected', 'pending_review'}:
            self.no_improvement = 0 if improved else self.no_improvement + 1
        if status == 'completed':
            self.small_gain_steps = self.small_gain_steps + 1 if delta_pp is not None and delta_pp < .1 else 0
        self.planning_failures = self.planning_failures + 1 if status == 'planning_failed' else 0
        self.stopped()
