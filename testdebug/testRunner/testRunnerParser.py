from ..helper.parseHelper import ArgProperty, ArgGroup, Parser, SUPPRESS

__all__ = ['TestRunnerParser']

class TestRunnerParser(Parser):
    def genDefaultArgGroups(self):
        # group of args relating to how the tests are executed
        testExecArgs = ArgGroup('testExecArgs',
            ArgProperty('-c', '--clean', action='store_true', help='Setting this flag deletes all files that have been automatically created by tests.'),
            ArgProperty('-d', '--debug', action='store_true', help='Setting this flag formats output from test failures in such a way as to ease inclusion of newly calculated values in future tests.'),
            ArgProperty('-f', '--failslow', action='store_true', help='Setting this flag will cause test execution to continue on errors. This will likely have the undesired effect of preventing breaking on fatal exception.'),
            ArgProperty('-p', '--profile', action='store_true', help='Setting this flag will cause all tests to be profiled. Only works when running with the default testing framework.'),
        )

        altFrameworkArgs = ArgGroup('altFrameworkArgs',
            ArgProperty('--pytest', action='store_true', help='Setting this flag will run all tests using the pytest framework instead of the unittest framework from the standard lib (need to have the pytest pkg installed)'),
            ArgProperty('--unittest', action='store_true', help='Setting this flag will run all tests using the unittest framework from the standard lib instead of the custom one in testBase'),
        )

        errorMsgArgs = ArgGroup('errorMsgArgs',
            ArgProperty('--delimiter', default=None, help='String used to separate the elements of containers (such as arrays) in error messages produced by tests.'),
            ArgProperty('--errorFmt', default=None, help='Format string (C-style) used for formatting data in error messages produced by tests. Currently only used for formatting array elements. Defaults to None, which implies autoformatting.'),
            ArgProperty('--truncate', default=None, help='Maximum number of elements of a single array that are shown in any error message produced by a test.'),
        )

        return [testExecArgs, altFrameworkArgs, errorMsgArgs]
