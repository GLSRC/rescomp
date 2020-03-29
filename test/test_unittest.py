# -*- coding: utf-8 -*-
""" Test if the unittest module works as it should """

import unittest

class BasicUnittestTesting(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_boolean_truth_value(self):
        true = True
        self.assertEqual(true, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
