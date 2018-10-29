class AsserterSkeleton(object):
    """Class that exposes skeletons of the 
    assert methods from unittest.TestCase
    """

    def assertFalse(self, expr, msg=None):
        pass

    def assertTrue(self, expr, msg=None):
        pass

    def assertRaises(self, expected_exception, *args, **kwargs):
        pass

    def assertWarns(self, expected_warning, *args, **kwargs):
        pass

    def assertLogs(self, logger=None, level=None):
        pass

    def assertEqual(self, first, second, msg=None):
        pass

    def assertNotEqual(self, first, second, msg=None):
        pass

    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        pass

    def assertNotAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        pass

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        pass

    def assertListEqual(self, list1, list2, msg=None):
        pass

    def assertTupleEqual(self, tuple1, tuple2, msg=None):
        pass

    def assertSetEqual(self, set1, set2, msg=None):
        pass

    def assertIn(self, member, container, msg=None):
        pass

    def assertNotIn(self, member, container, msg=None):
        pass

    def assertIs(self, expr1, expr2, msg=None):
        pass

    def assertIsNot(self, expr1, expr2, msg=None):
        pass

    def assertDictEqual(self, d1, d2, msg=None):
        pass

    def assertDictContainsSubset(self, subset, dictionary, msg=None):
        pass

    def assertCountEqual(self, first, second, msg=None):
        pass

    def assertMultiLineEqual(self, first, second, msg=None):
        pass

    def assertLess(self, a, b, msg=None):
        pass

    def assertLessEqual(self, a, b, msg=None):
        pass

    def assertGreater(self, a, b, msg=None):
        pass

    def assertGreaterEqual(self, a, b, msg=None):
        pass

    def assertIsNone(self, obj, msg=None):
        pass

    def assertIsNotNone(self, obj, msg=None):
        pass

    def assertIsInstance(self, obj, cls, msg=None):
        pass

    def assertNotIsInstance(self, obj, cls, msg=None):
        pass

    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):
        pass

    def assertWarnsRegex(self, expected_warning, expected_regex, *args, **kwargs):
        pass

    def assertRegex(self, text, expected_regex, msg=None):
        pass

    def assertNotRegex(self, text, unexpected_regex, msg=None):
        pass
