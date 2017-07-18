"""
Basic tester for py-pomdp.

author: mbforbes
"""

# builtins
import os
import sys
import unittest

# 3rd party
import numpy as np

# local
import pomdp


class POMDPEnvTest(unittest.TestCase):
    """Tests loading the POMDP Environment."""

    def setUp(self):
        """Load the POMDP Environment before each test.

        This is a bit excessive, but it's quick and eliminates shared
        state across tests.
        """
        # Load pomdp
        pomdpfile = "examples/env/env_parser_test.pomdp"
        # print 'Loading POMDP environment from', pomdpfile
        self.mypomdp = pomdp.POMDPEnvironment(pomdpfile)

        # Values
        self.testdiscount = 0.95
        self.testvalues = 'reward'
        self.teststates = ['heavy', 'light', 'novel']
        self.testactions = ['ask', 'sayHeavy', 'sayLight', 'sayNovel']
        self.testobservations = ['hearHeavy', 'hearLight', 'hearNovel']

    def test_headers(self):
        """Tests that the header values are as expected."""
        self.assertEqual(self.mypomdp.discount, self.testdiscount)
        self.assertEqual(self.mypomdp.values, self.testvalues)
        self.assertEqual(self.mypomdp.states, self.teststates)
        self.assertEqual(self.mypomdp.actions, self.testactions)
        self.assertEqual(self.mypomdp.observations, self.testobservations)

    def test_transitions(self):
        """Check transition values.

        Ask should be identity; all other actions should have the same
        vals.
        """
        for i in range(len(self.testactions)):
            if i == 0:
                # ask action
                for j in range(len(self.teststates)):
                    for k in range(len(self.teststates)):
                        expect = 1 if j == k else 0
                        self.assertEqual(self.mypomdp.T[(i, j, k)], expect)
            else:
                # all other actions
                for j in range(len(self.teststates)):
                    self.assertEqual(self.mypomdp.T[(i, j, 0)], 0.4)
                    self.assertEqual(self.mypomdp.T[(i, j, 1)], 0.4)
                    self.assertEqual(self.mypomdp.T[(i, j, 2)], 0.2)

    def test_observations(self):
        """Check observation values.

        Ask should have hand-tuned values; others should be uniform.
        """
        for i in range(len(self.testobservations)):
            if i == 0:
                # O: ask
                # 0.7 0.01 0.29
                # 0.01 0.7 0.29
                # 0.1 0.1 0.8
                self.assertEqual(self.mypomdp.Z[(0, 0, 0)], 0.7)
                self.assertEqual(self.mypomdp.Z[(0, 0, 1)], 0.01)
                self.assertEqual(self.mypomdp.Z[(0, 0, 2)], 0.29)
                self.assertEqual(self.mypomdp.Z[(0, 1, 0)], 0.01)
                self.assertEqual(self.mypomdp.Z[(0, 1, 1)], 0.7)
                self.assertEqual(self.mypomdp.Z[(0, 1, 2)], 0.29)
                self.assertEqual(self.mypomdp.Z[(0, 2, 0)], 0.1)
                self.assertEqual(self.mypomdp.Z[(0, 2, 1)], 0.1)
                self.assertEqual(self.mypomdp.Z[(0, 2, 2)], 0.8)
            else:
                # all other actions
                expect = 1.0 / float(len(self.testobservations))
                for j in range(len(self.teststates)):
                    for k in range(len(self.testobservations)):
                        assert self.mypomdp.Z[(i, j, k)] == expect

    def test_rewards(self):
        """Check the rewards are as expected."""
        # R: ask : * : * : * -1
        for i in range(len(self.teststates)):
            for j in range(len(self.teststates)):
                for k in range(len(self.testobservations)):
                    assert self.mypomdp.R[(0, i, j, k)] == -1

        # R: sayHeavy : heavy : * : * 5
        # R: sayHeavy : light : * : * -10
        # R: sayHeavy : novel : * : * -2
        for j in range(len(self.teststates)):
            for k in range(len(self.testobservations)):
                self.assertEqual(self.mypomdp.R[(1, 0, j, k)], 5)
                self.assertEqual(self.mypomdp.R[(1, 1, j, k)], -10)
                self.assertEqual(self.mypomdp.R[(1, 2, j, k)], -2)

        # R: sayLight : heavy : * : * -10
        # R: sayLight : light : * : * 5
        # R: sayLight : novel : * : * -2
        for j in range(len(self.teststates)):
            for k in range(len(self.testobservations)):
                self.assertEqual(self.mypomdp.R[(2, 0, j, k)], -10)
                self.assertEqual(self.mypomdp.R[(2, 1, j, k)], 5)
                self.assertEqual(self.mypomdp.R[(2, 2, j, k)], -2)

        # R: sayNovel : light : * : * -2
        # R: sayNovel : heavy : * : * -2
        # R: sayNovel : novel : * : * 5
        for j in range(len(self.teststates)):
            for k in range(len(self.testobservations)):
                self.assertEqual(self.mypomdp.R[(3, 0, j, k)], -2)
                self.assertEqual(self.mypomdp.R[(3, 1, j, k)], -2)
                self.assertEqual(self.mypomdp.R[(3, 2, j, k)], 5)


class POMDPEndToEndTest(unittest.TestCase):
    """Tests the loading of a 'full' POMDP (environment and policy),
    performs belief updates, and gets expected rewards."""

    def setUp(self):
        """Load the 'full' POMDP (environment and policy) before each
        test.
        """
        # Load pomdp
        self.pomdp = pomdp.POMDP(
            'examples/env/voicemail.pomdp',  # env
            'examples/policy/voicemail.policy',  # policy
            np.array([[0.65], [0.35]])  # prior
        )

        # How close our numbers have to be to pass the test.
        self.float_epsilon = 0.01

    def assertFloatsWithinEpsilon(self, a, b):
        """Asserts that two floats are within epsilon of eachother.

        Args:
            a (float)
            b (float)
            epsilon (float): How close a and b can be
        """
        self.assertTrue(
            a - self.float_epsilon <= b if a > b
            else b - self.float_epsilon <= a)

    def test_belief_updates(self):
        """Provide observations and do belief updates. Check:
            - actions
            - expected reward
            - belief
        """
        expected_actions = ['ask', 'ask', 'ask', 'doSave']
        expected_rewards = [3.46, 2.91, 3.13, 5.14]
        expected_beliefs = [
            np.array([.35, .65]),
            np.array([.59, .41]),
            np.array([.79, .21]),
            np.array([.65, .35])
        ]

        observations = ['hearDelete', 'hearSave', 'hearSave']
        obs_idx = 0
        best_action_str = None
        while True:
            # Get action and expcted rewards
            best_action_num, expected_reward = self.pomdp.get_best_action()
            best_action_str = self.pomdp.get_action_str(best_action_num)

            # Check action and expcted rewards
            self.assertEqual(best_action_str, expected_actions[obs_idx])
            self.assertFloatsWithinEpsilon(
                expected_reward, expected_rewards[obs_idx])

            # Check action.
            if best_action_str != 'ask':
                # We have a 'terminal' action (either 'doSave' or
                # 'doDelete')
                break
            else:
                # The action is 'ask': Provide our next observation.
                obs_str = observations[obs_idx]
                obs_idx += 1
                obs_num = self.pomdp.get_obs_num(obs_str)
                self.pomdp.update_belief(best_action_num, obs_num)

                belief = np.round(self.pomdp.belief.flatten(), 3)
                # Check beliefs
                for idx, b in enumerate(belief):
                    self.assertFloatsWithinEpsilon(
                        b, expected_beliefs[obs_idx - 1][idx])

        # Take the 'terminal' action, and beliefs should be reset to
        # prior.
        best_action_num, expected_reward = self.pomdp.get_best_action()
        # Observation doesn't affect this action.
        self.pomdp.update_belief(
            best_action_num, self.pomdp.get_obs_num('hearSave'))
        # Check beliefs redux
        belief = np.round(self.pomdp.belief.flatten(), 3)
        for idx, b in enumerate(belief):
            self.assertFloatsWithinEpsilon(b, expected_beliefs[-1][idx])

    def test_dumps(self):
        """Extremely basic test to ensure that belief / overview
        printing (dumping) don't crash.

        (Ahem... they've been previously emperically verified...)
        """
        # Stop from printing
        sys.stdout = open(os.devnull, "w")

        # Do the dumps.
        self.pomdp.belief_dump()
        self.pomdp.pomdpenv.print_summary()

        # Re-enable printing.
        sys.stdout = sys.__stdout__

if __name__ == '__main__':
    unittest.main()
