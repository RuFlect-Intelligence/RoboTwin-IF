"""Shared online/replay timer for continuously observed task success."""


class SuccessDuration:
    def __init__(self, task, stability):
        self.required_steps = stability["minimum_settle_steps"] + stability["observation_window_steps"]
        predicate = task.get("checker_config", {}).get("predicate", {})
        self.pick = predicate.get("name") == "pick_success"
        args = predicate.get("arguments", {})
        self.arm = args.get("requested_arm") or args.get("arm")
        self.subject = args.get("subject")
        self.reference_step = None
        self.candidate_since = None

    def observe(self, grippers, step):
        """Consume every physics step, including gaps between success checks.

        A new hold is a new continuous interval even if its relative drift is
        zero. The recorded reference step makes that identity replayable.
        """
        if not self.pick:
            return
        closure = grippers.get(self.arm, {}).get("closure_evidence", {})
        reference = None
        if closure.get("valid") and closure.get("subject") == self.subject:
            reference = closure.get("reference_step")
            if type(reference) is not int or not 0 < reference <= step:
                raise ValueError("Pick hold reference step missing or invalid")
        if reference is None or reference != self.reference_step:
            self.candidate_since = None
        self.reference_step = reference

    def evaluate(self, result, step):
        if result["passed"] and (not self.pick or self.reference_step is not None):
            if self.candidate_since is None:
                self.candidate_since = step
        else:
            self.candidate_since = None
        elapsed = 0 if self.candidate_since is None else step - self.candidate_since
        if elapsed < self.required_steps:
            return {**result, "status": "failed", "passed": False,
                    "failed_predicates": [*result.get("failed_predicates", []),
                                          "policy_observed_settle_duration"],
                    "observed_settle_steps": elapsed,
                    "required_settle_steps": self.required_steps}
        return result
