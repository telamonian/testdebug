import errno
import os
from pathlib import Path
import random
import shutil
from six import print_
import sys
import string
import tarfile
import time
import traceback

from .asserter import Asserter

__all__ = ['TestBase', 'color']

class _TextColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def color(cls, color, s):
        colorCode = getattr(cls, color)
        return colorCode + s + cls.ENDC
color = _TextColor.color

class TestBase(Asserter):
    # the dir where test data is stored and tmp test files are written to
    testDirPath = None

    # the file or list of files to read from prior to the test
    testFilePath = None

    # the path to use when writing out a temporary test file
    testOutputPath = None

    # default values for things to stick on the end of temporary test file names
    testOutputDelimiter = '_-_'
    testOutputSuffix = 'writeTest'

    # flag to indicate that a temporary test file has been written
    testOutputWritten = False

    # path to an archive containing test data, if any. The tar should following the naming convention that foo.tar.gz unpacks to foo. The archive will automatically be unpacked and cleaned up as appropriate
    # NB: be careful! Any dir/file foo that happens to be in the directory when we unpack/clean up foo.tar.gz will be deleted
    testTarPath = None

    # set if/when the tarfile at cls.testTarPath is unpacked
    testTarUnpackedFilesPath = None

    @staticmethod
    def getTestOutputPath(testOutputPath, outputDelimiter=None, outputSuffix=None, randPath=None):
        if outputDelimiter is None: outputDelimiter = TestBase.testOutputDelimiter
        if outputSuffix is None: outputSuffix = TestBase.testOutputSuffix
        if randPath is None: randPath = True

        p = Path(testOutputPath)

        # add suffix to output path, if requested
        if outputSuffix:
            p = p.with_name(outputDelimiter.join((p.stem, outputSuffix)) + p.suffix)

        # randomize the output path, if requested
        if randPath:
            p = TestBase.randomizedPath(p)

        return p

    @staticmethod
    def randStr(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    @staticmethod
    def randomizedPath(p, size=6, chars=string.ascii_uppercase + string.digits):
        p = Path(p)
        return p.with_name(p.stem + TestBase.randStr(size=size, chars=chars) + p.suffix)

    @classmethod
    def getTestNames(cls):
        """Get the test methods names from the tests class attr if it exists, or find them in the cls based on their names having the 'test' prefix.
        """
        try:
            for testName in cls.tests:
                yield testName
        except AttributeError:
            for testName in (testName for testName in dir(cls) if testName[:4]=='test' and callable(getattr(cls, testName))):
                yield testName

    @classmethod
    def runSimple(cls, failslow=False):
        """Use this method if you just want to execute your tests as normal Python code rather than unittests within some framework

        Required for catching fatal exceptions when using a debugger (these are apparently swallowed by unittest and its brethren)
        """
        # add an excepthook that flushes any tester output and then pauses for a tic (to finish the flush)
        old_excepthook = sys.excepthook
        def excepthook(type, value, traceback):
            sys.stdout.flush()
            time.sleep(.05)

            old_excepthook(type, value, traceback)
        sys.excepthook = excepthook

        # modify stderr.write so that progress bars won't clobber tester output
        # TODO: simplify, expand to deal with stdout, output from things other than progress bars, etc
        stderrWrite = sys.stderr.write
        def stderrWriteWithConvert(*args, **kwargs):
            if stderrWriteWithConvert.doConvert:
                if args[0].startswith('\r'):
                    args = ('\n' + args[0],) + args[1:]
                    stderrWriteWithConvert.didConvert = True
                stderrWriteWithConvert.doConvert = False

            return stderrWrite(*args, **kwargs)
        stderrWriteWithConvert.doConvert = False
        stderrWriteWithConvert.didConvert = False
        sys.stderr.write = stderrWriteWithConvert
        
        print_('running test class: %s' % color('HEADER', cls.__name__), flush=True)

        # test class start time
        testClassStartTime = time.time()
        testPassCount = 0
        testFailCount = 0

        obj = cls()

        # per test class setup/teardown via context manager
        with cls.setUpClassContext():
            for testName in obj.getTestNames():
                testMethod = obj.__getattribute__(testName)

                print_('\t%-60s' % (color('OKBLUE', testName) + '...'), end='', flush=True)
                stderrWriteWithConvert.doConvert = True

                # test case start time
                testCaseStartTime = time.time()

                try:
                    # per test case setup/teardown via context manager
                    with obj:
                        # the actual test code
                        testMethod()

                    extraSpace = '\t\t' if stderrWriteWithConvert.didConvert else ''
                    print_(color('OKGREEN', extraSpace + 'passed') + ', %10.3f seconds' % (time.time() - testCaseStartTime), flush=True)
                    testPassCount += 1
                except Exception as e:
                    extraSpace = '\t\t' if stderrWriteWithConvert.didConvert else ''
                    print_(color('FAIL', extraSpace + 'failed') + ', %10.3f seconds' % (time.time() - testCaseStartTime), end='\n\n', flush=True)
                    if not failslow:
                        raise e
                    else:
                        print_(color('WARNING', traceback.format_exc()))
                        testFailCount += 1
                finally:
                    stderrWriteWithConvert.didConvert = False

        totalTestTime = time.time() - testClassStartTime
        print_('total %.3f seconds\n' % (totalTestTime))

        return testPassCount,testFailCount,totalTestTime

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def __enter__(self):
        self.setUp()

    def __exit__(self, excType, excValue, traceback):
        self.tearDown()

    #### stuff for enabling the ez creation and teardown of test files
    @classmethod
    def setUpClass(cls):
        # untar archived test data, if any
        if cls.testTarPath is not None:
            testTar = tarfile.open(str(cls.testTarPath))

            # assume that the root dir of the first entity in the tarfile is the containing directory for everything in the tarfile
            cls.testTarUnpackedFilesPath = cls.testTarPath.parent / testTar.getnames()[0].split('/')[0]

            # extract the tarfile
            testTar.extractall(str(cls.testTarPath.parent))

        # write out a test file if the .writeTestFile cls method has been overridden in a subclass
        cls.testOutputWritten = cls.writeTestFile(testOutputPath=cls.testOutputPath)

    @classmethod
    def tearDownClass(cls):
        # clean up the test file written at the start of the test suite, if any
        if cls.testOutputWritten:
            cls.cleanTestFile(testOutputPath=cls.testOutputPath)

        # if any archived test data was unpacked, clean that up
        if cls.testTarUnpackedFilesPath is not None:
            # remove the whole tree of the unpacked tarfile
            shutil.rmtree(str(cls.testTarUnpackedFilesPath))

            # for good measure
            cls.testTarUnpackedFilesPath = None

    @classmethod
    def setUpClassContext(cls):
        """Returns a context manager that runs setUpClass on enter and tearDownClass on exit
        """
        class SetUpClassContext(object):
            def __enter__(self):
                cls.setUpClass()

            @classmethod
            def __exit__(self, excType, excValue, traceback):
                cls.tearDownClass()

        return SetUpClassContext()

    @classmethod
    def cleanAll(cls, testOutputRoot, dryrun=True, verbose=True):
        """Removes all files whose name ends in testOutputDelimiter + testOutputSuffix recursively in testOutputRoot
        """
        for pth in (Path(p)/Path(f) for p,ds,fs in os.walk(testOutputRoot) for f in fs):
            if pth.stem.split(cls.testOutputDelimiter)[-1].startswith(cls.testOutputSuffix):
                if dryrun:
                    if verbose:
                        print('dryrun, would have removed: %s' % pth)
                else:
                    pth.unlink()

                    if verbose:
                        print('removed: %s' % pth)

    @classmethod
    def cleanTestFile(cls, testOutputPath, doRaise=False):
        try:
            Path(str(testOutputPath)).unlink()
        except OSError as e:
            # errno.ENOENT = no such file or directory
            if doRaise or e.errno != errno.ENOENT:
                raise e

    @classmethod
    def writeTestFile(cls, testOutputPath):
        """Child versions should return True if a test file is successfully written out
        """
        return False
