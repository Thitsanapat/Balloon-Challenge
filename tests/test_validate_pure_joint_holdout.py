import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_pure_joint_holdout import read_single_flight


class PureJointHoldoutFormatTests(unittest.TestCase):
    def test_one_flight_list_is_unwrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps([{'seed': 17, 'score': 4}]))
            self.assertEqual(read_single_flight(path)['score'], 4)

    def test_object_or_multiple_flights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            for payload in ({'seed': 17}, [{'seed': 17}, {'seed': 18}]):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    read_single_flight(path)


if __name__ == '__main__':
    unittest.main()
