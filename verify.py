"""Preparation/collection checks; analysis verification belongs to its owner."""
import sys
import unittest
sys.dont_write_bytecode=True

def main():
    suite=unittest.TestSuite()
    for area in ('preparation/tests','benchmarking/tests'):
        suite.addTests(unittest.defaultTestLoader.discover(area,pattern='test_*.py',top_level_dir='.'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__': raise SystemExit(main())
