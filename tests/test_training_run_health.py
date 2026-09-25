import unittest
from BalloonPoppingGymEnv.training.run_health import TruncationGuard


class HealthTests(unittest.TestCase):
    def test_repeated_truncations_stop_and_complete_episodes_reset_streak(self):
        guard = TruncationGuard(3)
        self.assertFalse(guard.observe(True))
        self.assertFalse(guard.observe(True))
        self.assertFalse(guard.observe(False))
        self.assertFalse(guard.observe(True))
        self.assertFalse(guard.observe(True))
        self.assertTrue(guard.observe(True))
        self.assertEqual(guard.total, 5)

    def test_invalid_limit(self):
        for value in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                TruncationGuard(value)


if __name__ == '__main__':
    unittest.main()
