import gzip
import json
from pathlib import Path
import tempfile
import unittest

from robotwin_if.grounding import score_trace


class GroundingTests(unittest.TestCase):
    def score(self, pool, physical, settle):
        subject = "object"
        node = {"pool": pool, "task": {"goal": {"subject": subject, "relation": "left_of"}}}
        result = {"initial_state_hash": "state", "initial_observation_hash": "obs", "physics_steps": 80}
        initial = {
            "kind": "initial", "dt": .01, "physical_state_hash": "state", "observation_hash": "obs",
            "success_initial": {"entities": {subject: {"support_plane_z": 0.0}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gz"
            with gzip.open(path, "wt") as out:
                out.write(json.dumps(initial) + "\n")
                for step in range(1, 81):
                    out.write(json.dumps({"kind": "physics", "step": step, **physical(step)}) + "\n")
                    if step % 20:
                        continue
                    height, geometry = settle(step)
                    entity = {"aabb": {"min": [0., 0., height]}, "support_plane_z": 0.}
                    checks = {"relation_surface_gap_band": geometry,
                              "relation_directional_orthogonal_gap": geometry,
                              "relation_entities_stable": False}
                    row = {"kind": "success_check", "step": step,
                           "snapshot": {"terminal": {"entities": {subject: entity}}},
                           "result": {"metrics": {"relation_holds": {
                               "status": "failed", "metrics": {"checks": checks}}}}}
                    out.write(json.dumps(row) + "\n")
            return score_trace(path, node, result)

    def test_pick_requires_gripper_contact_strictly_above_threshold(self):
        contacts = [
            {"arm": "left", "entity_a": "table", "entity_b": "object", "summed_impulse_norm": 1.},
            {"arm": "left", "entity_a": "gripper:left", "entity_b": "object", "summed_impulse_norm": 1e-6},
        ]
        settle = lambda step: (0., True)
        self.assertFalse(self.score("pick", lambda step: {"contacts": contacts}, settle)["passed"])
        actual = {**contacts[-1], "summed_impulse_norm": 1e-5}
        result = self.score("pick", lambda step: {"contacts": [actual] if step == 10 else contacts}, settle)
        self.assertEqual(result["first_step"], 10)

    def test_placement_requires_acquisition_and_contiguous_near_surface_dwell(self):
        closure = {"valid": True, "subject": "object", "path": "contact", "actual_width_m": .01}
        gripper = {"closure_evidence": closure, "actual_width_m": .01}
        acquired = lambda step: {"actual_grippers": {"right": gripper} if step == 3 else {}}
        self.assertFalse(self.score("d2b", lambda step: {"actual_grippers": {}},
                                    lambda step: (.01, True))["passed"])
        self.assertFalse(self.score("d2b", acquired, lambda step: (.03 if step == 40 else .01, True))["passed"])
        self.assertFalse(self.score("d2b", acquired, lambda step: (.01, step != 40))["passed"])
        result = self.score("d2b", acquired, lambda step: (.01, True))
        self.assertEqual(result, {"passed": True, "first_step": 80, "acquisition_step": 3})


if __name__ == "__main__":
    unittest.main()
