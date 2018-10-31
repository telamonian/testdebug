#!/usr/bin/env python

##########################
#### external imports ####
##########################

import inspect
import os
from pathlib import Path
from six import print_
import sys
import time
import unittest

#########################
#### project imports ####
#########################

from ..helper import importHelper as impHlp, profileHelper as profHlp
from ..testBase import TestBase, color
from .testRunnerParser import TestRunnerParser

#####################################################
#### uncomment below to treat warnings as errors ####
#####################################################

# import warnings
# warnings.filterwarnings('error')
# warnings.filterwarnings('error', category=FutureWarning)

########################
#### numpy settings ####
########################

# np.set_printoptions(precision=1, threshold=1e6, linewidth=1e6)
# np.seterr(all='warn')

##########################
#### application code ####
##########################

class TestRunner(object):
    helpMessage = 'A script to simplify running/debugging Python unittests.'

    debugErrorMsgKwargs = dict([
        ('delimiter', ','),
        ('errorFmt',  None),
        ('truncate',  int(1e6)),
    ])

    def __init__(self, root=None, localsDict=None, varsDict=None, disableCleanUpInt=True):
        self.root = root
        self.localsDict = localsDict
        self.varsDict = varsDict

        _frame = inspect.currentframe().f_back
        try:
            _module = inspect.getmodule(_frame)
            if self.root is None: self.root = Path(os.path.dirname(os.path.abspath(_module.__file__ ))) if root is None else Path(root)
            if self.localsDict is None: self.localsDict = _frame.f_locals
            if self.varsDict is None: self.varsDict = vars(_module)
        finally:
            del _frame

        self.parser = TestRunnerParser(description=self.helpMessage)

        self.disableCleanUpInt = disableCleanUpInt
        self.testBaseDict = {}

    def getTestBases(self):
        # get the TestBases and group them by module
        tbByMod = {}
        for tb in (tb for tb in self.varsDict.values() if inspect.isclass(tb) and tb.__name__[-4:]=='Base'):
            try:
                tbByMod[tb.__module__].append(tb)
            except KeyError:
                tbByMod[tb.__module__] = [tb]

        # order by module order given in sys.modules (which itself is ordered in CPython >= 3.6)
        return [tb for modName in (m for m in sys.modules if m in tbByMod) for tb in sorted(tbByMod[modName], key=lambda x: x.__name__)]

    def getTestCases(self, exportToLocals=False):
        testCases = []
        for testBase in self.getTestBases():
            testCaseName = testBase.__name__[:-4] + 'Case'
            TestCase = type(testCaseName, (testBase, unittest.TestCase), {})
            
            if self.disableCleanUpInt:
                TestCase.disableCleanUpInt = True
            if exportToLocals:
                self.localsDict[testCaseName] = TestCase
            
            testCases.append(TestCase)
        return testCases

    def main(self, **kwargs):
        self.parser.run()

        self.discoverInPath(self.parser['path'])

        if self.parser['clean']:
            self.clean()
        elif self.parser['pytest']:
            self.runPytest(**kwargs)
        elif self.parser['unittest']:
            self.runUnittest(failslow=self.parser['failslow'], **kwargs)
        else:
            errorMsgKwargs = self.parser.getArgDict(group='errorMsgArgs')
            if self.parser['debug']:
                errorMsgKwargs.update(self.debugErrorMsgKwargs)

            self.runSimple(errorMsgKwargs=errorMsgKwargs,
                           failslow=self.parser['failslow'],
                           profile=self.parser['profile'],
                           **kwargs)

    def clean(self):
        TestBase.cleanAll(dryrun=False, verbose=True)

    def discoverInFullname(self, fullname):
        return self.testBaseDict.update(impHlp.DeepImport(fullname=fullname, attrs=['*'], attrFilter='Base$', locals=self.localsDict))

    def discoverInPath(self, path):
        for pth in (Path(pth) for pth in path):
            try:
                # use name parts of pth relative to self.root
                name = '.'.join(pth.relative_to(self.root).with_suffix('').parts)
            except ValueError:
                if not pth.is_absolute():
                    # use name parts from a relative path
                    name = '.'.join(pth.with_suffix('').parts)
                else:
                    # leave name blank
                    name = None

            self.testBaseDict.update(impHlp.DeepImport(path=pth, name=name, attrs=['*'], attrFilter='Base$', locals=self.localsDict))

        return self.testBaseDict

    def runPytest(self, **kwargs):
        import pytest

        thisScriptPath = os.path.realpath(__file__)
        pytest.main(args= thisScriptPath + ' -s', **kwargs)

    def runUnittest(self, failslow=False, **kwargs):
        failfast = not failslow

        self.getTestCases(exportToLocals=True)
        unittest.main(failfast=failfast, **kwargs)
    
    def runSimple(self, errorMsgKwargs=None, failslow=False, profile=False, **kwargs):
        if errorMsgKwargs is None: errorMsgKwargs = {}

        start = time.time()
        testPassCount,testFailCount,testCaseCount = 0,0,0

        for TestCase in self.getTestCases():
            TestCase.setErrorMsgKwargs(**errorMsgKwargs)

            if profile:
                testCasePassCount,testCaseFailCount,testCaseTime = profHlp.ProfileDecorator()(TestCase.runSimple)(failslow=failslow)
            else:
                testCasePassCount,testCaseFailCount,testCaseTime = TestCase.runSimple(failslow=failslow)

            testPassCount += testCasePassCount
            testFailCount += testCaseFailCount
            testCaseCount += 1

        print_(color('OKGREEN',                              '%d tests passed' % testPassCount))
        print_(color('FAIL' if testFailCount else 'OKGREEN', '%d tests failed' % testFailCount))
        print_(                                              '%d test cases finished, %.3f seconds' % (testCaseCount, time.time() - start))

if __name__ == '__main__':
    testRunner = TestRunner(varsDict=vars(), localsDict=locals(), disableCleanUpInt=True)
    testRunner.main()

    # create the TestCases and export them to this module's namespace
    # testRunner.getTestCases(exportToLocals=True)
