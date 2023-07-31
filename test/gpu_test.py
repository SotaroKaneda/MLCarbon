
import unittest

class Gpu_test(unittest.TestCase):
    def test(self):
        one = 1
        two = 2
        self.assertEqual(two - one, 1, "incorrect math")

unittest.main()