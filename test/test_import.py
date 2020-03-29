# -*- coding: utf-8 -*-
""" Tests if the rescomp modules can be imported as intended """

import unittest
import numpy as np
from rescomp import measures

class testImport(unittest.TestCase):
    def test_import(self):
        import rescomp
        import rescomp.esn
        import rescomp.measures
        import rescomp.utilities
        import rescomp.simulations

if __name__ == "__main__":
    unittest.main(verbosity=2)
