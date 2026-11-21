"""Event and finite-search regression fixtures; runnable with unittest."""
import unittest

import numpy as np

from disturbances import DISTURBANCE, WINDOW, Event, PolicyResult, Scenario, _abort, replay, run_scenario
from event_checks import separation_audit, verify_history
from exact_schedule import solve_exact
from geometry import ANGULAR_SPEED, OBSERVE, SPEED, STARTS, TARGETS, Router, Target
from scheduler import Mission, Space, State, evaluate_fixed, make_leg


class EventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = Router()
        cls.targets = {target.name: target for target in TARGETS}
        cls.layout = [
            ("T1", (2, 4), [[15.011163, 87.110158], STARTS[4]]),
            ("T2", (0, 3), [STARTS[0], [87.994645, 70.804645]]),
            ("T3", (2, 1), [[16.711249, 88.758751], STARTS[1]]),
        ]
        cls.plan = evaluate_fixed(cls.router, cls.targets, cls.layout, STARTS, (0.,) * len(STARTS))
        if cls.plan is None:
            raise ValueError("The event test fixture requires a feasible nominal plan.")

    def test_failure_during_observation_locks_it_and_releases_healthy_vehicle_after_reset(self):
        first = self.plan[0]
        scenario = Scenario("failure", "failure", failure=4)
        events, locked, _, release, _ = replay(self.plan, scenario)
        self.assertEqual(locked, {"T1": first.start})
        partner = next(leg for leg in first.legs if leg.vehicle == 2)
        self.assertAlmostEqual(release[2], first.start + OBSERVE + partner.slew)
        self.assertTrue(np.isinf(release[4]))
        observations = {e.vehicle: e for e in events if e.phase == "observe"}
        self.assertAlmostEqual(observations[4].end, DISTURBANCE)
        self.assertAlmostEqual(observations[2].end, first.start + OBSERVE)

    def test_partner_wait_and_its_next_mission_are_blocked(self):
        layout = [*self.layout[:2], ("T3", (0, 1), STARTS[[0, 1]])]
        plan = evaluate_fixed(self.router, self.targets, layout, STARTS, (0.,) * len(STARTS))
        scenario = Scenario("wait", "failure+outage", failure=3, recovery=10.)
        events, locked, _, _, engaged = replay(plan, scenario)
        self.assertEqual(set(locked), {"T1"})
        name, leg = engaged[0]
        self.assertEqual(name, "T2")
        self.assertAlmostEqual(leg.ready, 10.)
        self.assertTrue(any(e.vehicle == 0 and e.phase == "wait_sync" and e.end == 10. for e in events))
        through, starts, *_ = replay(plan, scenario, through_window=True)
        self.assertEqual(set(starts), {"T1"})
        self.assertFalse(any(e.vehicle == 0 and e.target == "T3" for e in through))
        self.assertTrue(any(e.vehicle == 0 and e.phase == "wait_sync" and e.end == WINDOW for e in through))

    def test_abort_mid_slew_uses_the_actual_gimbal_angles(self):
        scenario = Scenario("slew", "new", new=Target("new", 0., 50., 50.))
        _, _, _, _, engaged = replay(self.plan, scenario)
        original = next(leg for mission in self.plan if mission.target == "T2" for leg in mission.legs if leg.vehicle == 3)
        _, current = engaged[3]
        self.assertAlmostEqual(current.initial_pan, ANGULAR_SPEED * (DISTURBANCE - original.arrive))
        self.assertAlmostEqual(current.initial_tilt, original.tilt)
        available, actions = _abort(3, engaged[3], DISTURBANCE)
        self.assertAlmostEqual(available, DISTURBANCE + abs(current.initial_pan) / ANGULAR_SPEED)
        self.assertEqual(actions[-1].phase, "abort_reset")
        self.assertLess(actions[-1].end - actions[-1].begin, original.slew)

    def test_repair_reuses_one_vehicles_pointing_after_replacing_its_partner(self):
        stations = np.array([leg.path[-1] for mission in self.plan for leg in mission.legs])
        space = Space(self.router, TARGETS, None, np.vstack((STARTS, stations)))
        scenario = Scenario("replace_partner", "failure+outage", failure=3, recovery=10.)
        policies = run_scenario(space, self.plan, scenario)
        repair = policies["repair"]
        self.assertIn((0, "T2"), repair.continued)
        self.assertFalse(any(e.vehicle == 0 and e.phase == "abort_reset" for e in repair.events))
        self.assertEqual({e.vehicle for e in repair.events if e.phase == "observe" and e.target == "T2"}, {0, 4})
        verify_history(policies["retain"], repair, scenario)

    def test_redirect_during_transfer_starts_at_the_interpolated_position(self):
        original = next(leg for mission in self.plan if mission.target == "T3" for leg in mission.legs if leg.vehicle == 2)
        cutoff = (original.depart + original.arrive) / 2.
        _, _, positions, _, engaged = replay(self.plan, Scenario("travel", "outage", recovery=cutoff))
        expected = (original.path[0] + original.path[-1]) / 2.
        np.testing.assert_allclose(positions[2], expected, atol=1e-8)
        available, actions = _abort(2, engaged[2], cutoff)
        self.assertEqual(available, cutoff)
        self.assertTrue(all(e.phase != "abort_reset" for e in actions))
        redirected = make_leg(self.router, 2, positions[2], STARTS[2], available, self.targets["T1"])
        np.testing.assert_allclose(redirected.path[0], expected, atol=1e-8)
        self.assertAlmostEqual(redirected.depart, cutoff)

    def test_recovery_coincident_with_original_observation_locks_the_original_start(self):
        cutoff = next(m.start for m in self.plan if m.target == "T2")
        new = Target("new", 0., 50., 50.)
        scenario = Scenario("recovery_edge", "new+outage", new=new, recovery=cutoff)
        before, locked, _, _, _ = replay(self.plan, scenario)
        through, starts, *_ = replay(self.plan, scenario, through_window=True)
        self.assertEqual(set(locked), {"T1", "T2"})
        self.assertEqual(locked["T2"], cutoff)
        original = PolicyResult(starts, through, "retain")
        verify_history(original, PolicyResult(locked, before, "continued"), scenario)
        command = Event(1, new.name, "at_station", cutoff - .01, cutoff - .01, STARTS[1:2])
        with self.assertRaisesRegex(ValueError, "before communication"):
            verify_history(original, PolicyResult(locked, [*before, command], "premature"), scenario)

    def test_failure_precedes_a_new_observation_at_the_disturbance_instant(self):
        target = self.targets["T3"]
        legs = tuple(make_leg(self.router, v, STARTS[v], STARTS[v], 0., target) for v in (0, 3))
        mission = Mission(target.name, DISTURBANCE, legs)
        events, starts, *_ = replay([mission], Scenario("tie", "failure", failure=3), through_window=True)
        self.assertFalse(starts)
        self.assertFalse(any(e.phase == "observe" for e in events))

    def test_continuous_separation_detects_crossing_between_endpoints(self):
        crossing = np.array([55., 14.])
        finish = np.array([70., 14.])
        first = Event(0, "T1", "travel", 0., np.linalg.norm(STARTS[0] - finish) / SPEED,
                      np.array([STARTS[0], finish]))
        ray = crossing - STARTS[1]
        distance = np.linalg.norm(ray)
        meet = np.linalg.norm(crossing - STARTS[0]) / SPEED
        second = Event(1, "T3", "travel", meet - distance / SPEED, meet + 3.,
                       np.array([STARTS[1], crossing + 3. * SPEED * ray / distance]))
        result = separation_audit([first, second], first.end)
        self.assertLess(result["min_separation_m"], 1e-7)
        self.assertAlmostEqual(result["closest_time_s"], meet)
        self.assertEqual(result["travel_travel_pairs"], 1)


class ExactSearchTests(unittest.TestCase):
    def test_deadline_applies_to_synchronized_start_not_observation_end(self):
        router = Router()
        space = Space(router, (TARGETS[0],), None, STARTS[[2, 4]])
        initial = State(space.origins(STARTS), (np.inf, np.inf, 58., np.inf, 58.), latest=58.)
        before, early = solve_exact(space, initial, ("T1",), deadline=59.)
        within, late = solve_exact(space, initial, ("T1",), deadline=60.)
        self.assertTrue(early["confirmed_on_pool"] and late["confirmed_on_pool"])
        self.assertEqual(before.choices, ())
        self.assertEqual(len(within.choices), 1)
        self.assertLess(within.latest, 60.)
        self.assertGreater(within.latest + OBSERVE, 60.)

    def test_fixed_pair_can_be_kept_while_another_visit_is_inserted_before_it(self):
        router = Router()
        points = STARTS.copy()
        points[2] = (16.711249, 88.758751)
        space = Space(router, (TARGETS[0], TARGETS[2]), None, points[[1, 2, 4]])
        indices = space.origins(points)
        initial = State(indices, (np.inf, 0., 0., np.inf, 20.))
        fixed = {"T1": ((2, 4), (indices[2], indices[4]))}
        appended, serial = solve_exact(space, initial, ("T1", "T3"), fixed=fixed, retained_first=True)
        inserted, interleaved = solve_exact(space, initial, ("T1", "T3"), fixed=fixed)
        self.assertTrue(serial["confirmed_on_pool"] and interleaved["confirmed_on_pool"])
        self.assertEqual(len(inserted.choices), 2)
        self.assertEqual(tuple(c.target for c in inserted.choices), ("T3", "T1"))
        self.assertLess(inserted.latest, appended.latest)
        _, incomplete = solve_exact(space, initial, ("T1", "T3"), fixed=fixed, state_budget=1)
        self.assertFalse(incomplete["confirmed_on_pool"])


if __name__ == "__main__":
    unittest.main()
