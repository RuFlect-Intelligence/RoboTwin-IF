"""Either arm can satisfy the unchanged Pick checker."""

from copy import deepcopy

from if_benchmark.evaluation.success_duration import SuccessDuration
from .panel import require


class AnyArmPhysicalObserver:
    """One observed snapshot, two unchanged checkers, independent hold clocks.

    No handoff can concatenate two shorter holds into the required duration.
    Original task records and the default shared evaluator are never patched.
    """
    def __init__(self, env, task, source_hashes, *, observer_class=None, checker=None):
        if observer_class is None:
            from if_benchmark.evaluation.observation import PhysicalObserver
            observer_class = PhysicalObserver
        if checker is None:
            from if_benchmark.scene_validation.core import evaluate_task_checker
            checker = evaluate_task_checker
        require(task['goal']['verb'] == 'pick' and task['goal']['arm'] is None,
                'Only derived arm-neutral Pick is supported')
        self.checker = checker
        self.tasks = {}
        for arm in ('left', 'right'):
            derived = deepcopy(task)
            # Only checker binding varies. No arm becomes part of the instruction.
            derived['checker_config']['predicate']['arguments'].update(requested_arm=arm)
            derived['checker_config']['predicate']['arguments'].pop('arm', None)
            require(not any(a.get('axis') == 'D4' for a in
                derived.get('goal_provenance', {}).get('intended_axes', [])), 'D4 must be removed in derived task')
            self.tasks[arm] = derived
        self.base = observer_class(env, self.tasks['left'], source_hashes)
        self.right_timer = SuccessDuration(self.tasks['right'], self.base.policy)
        self.success_result = self.success_snapshot = self.success_step = self.success_arm = None
        self.last_snapshot = self.last_result = None

    def __getattr__(self, key):
        return getattr(self.base, key)

    def observe(self):
        contacts = self.base.observe()
        self.right_timer.observe(self.base.grippers, self.base.step)
        return contacts

    def evaluate(self):
        snapshot, left = self.base.evaluate()
        right = self.checker(self.tasks['right'], snapshot)
        require(right['status'] != 'blocked', 'Right-arm physical checker lacks required evidence')
        right = self.right_timer.evaluate(right, self.base.step)
        results = {'left':left, 'right':right}
        winner = next((a for a, result in results.items() if result['passed']), None)
        # Preserve the canonical result for evidence replay; arm-OR is explicit.
        chosen = results[winner or 'left']
        self.last_snapshot, self.last_result = snapshot, chosen
        self.last_per_arm = results
        if winner and self.success_result is None:
            self.success_result, self.success_snapshot = chosen, snapshot
            self.success_step, self.success_arm = self.base.step, winner
        return snapshot, chosen
