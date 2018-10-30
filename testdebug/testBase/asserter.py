from inspect import ismethod
import nppretty
import numpy as np
import operator
import unittest
from unittest.case import _AssertRaisesContext
from six import add_metaclass
import traceback
import types

from .asserterSkeleton import AsserterSkeleton

__all__ = ['Asserter']

def _coerceArray(array):
    """Basically the same thing as np.asanyarray, but also appropriately coerces Nonetypes and generators into arrays.
    """
    if array is None:
        return np.array([])
    elif isinstance(array, types.GeneratorType):
        return np.asanyarray(tuple(array))
    else:
        return np.asanyarray(array)

def _allWrap(func):
    """For wrapping test funcs with np.all, so as to guarantee that they return a single bool (and never an array of bools).
    """
    def allWrapped(*args, **kwargs):
        return np.all(func(*args, **kwargs))
    return allWrapped

def _npEqual(lhs, rhs):
    """Return False if np.testing.assert_equal would raise an Assertion error, and True otherwise.
    _npEqual(NaN, NaN) evaulates to True, as does _npEqual(inf, inf).
    """
    try:
        np.testing.assert_equal(lhs, rhs)
        return True
    except AssertionError:
        return False

class _AssertNotRaisesContext(_AssertRaisesContext):
    """a context manager used to implement the Asserter.assertNotRaises method.
    Inherits from unittest.case._AssertRaisesContext and inverts its .__exit__(...) method
    """
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            # store exception, without traceback, for later retrieval
            self.exception = exc_value.with_traceback(None)

            try:
                exc_name = self.expected.__name__
            except AttributeError:
                exc_name = str(self.expected)
            if self.obj_name:
                self._raiseFailure("{} raised by {}".format(exc_name,
                    self.obj_name))
            else:
                self._raiseFailure("{} raised".format(exc_name))
        else:
            traceback.clear_frames(tb)

        return True

class AsserterMetaclass(type):
    def __new__(cls, clsname, bases, dct):
        # add all of the asserts defined in unittest.TestCase that are not defined in Asserter
        dctWithAsserts = {k:v
                          for k,v in vars(unittest.TestCase).items()
                          if k.startswith('assert') and ismethod(v) and 'deprecated' not in v.__name__}
        dctWithAsserts.update(dct)

        return super(AsserterMetaclass, cls).__new__(cls, clsname, bases, dct)

@add_metaclass(AsserterMetaclass)
class Asserter(AsserterSkeleton):
    ## some defaults for formatting error messages
    delimiter = ','
    errorFmt = None
    truncate = int(1e3)

    @classmethod
    def setErrorMsgKwargs(cls, **errorMsgKwargs):
        allowedKeys = ('delimiter', 'errorFmt', 'truncate')

        for key,val in errorMsgKwargs.items():
            if key not in allowedKeys:
                raise ValueError('Tried to set an error message formatting class attr with an unknown key. key: %s, allowedKeys: %s' % (key, allowedKeys))

            if val is not None:
                # assume the default value is desired if the passed-in val is None
                setattr(cls, key, val)

    ## hand-rolled, fine-tuned assert methods
    def assertAlmostEqual(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False, msg=None):
        testBool = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

        if not testBool:
            if msg is None:
                msg = '%.12f is not close to %.12f.' % (a, b)

        self.assertTrue(testBool, msg)

    def assertAll(self, booleans, msg=None):
        """Assert that all entries in the sequence booleans eval to True.
        """
        testBool = all(booleans)

        if not testBool:
            # only bother formatting the msg str if we need to
            total = len(booleans)
            falses = np.where(np.logical_not(booleans))[0]
            falseCount = falses.size

            msg = 'Not all elements of booleans sequence were True.\n' \
                  '%d out of %d elements were False.\n'                \
                  'The False elements were at indices:\n%s\n'          \
                  'booleans:\n%s\n' % (total, falseCount, nppretty.formatArrayAsArray(falses), booleans)

        self.assertTrue(testBool, msg=msg)

    def _assertArray(self, intent, actual, addbrackets, delimiter, errorFmt, msg, newline,
                     squeeze, truncate, testFunc, testFuncArgs=None, testFuncKwargs=None, **kw):
        """Perform an assertion on two arrays using a passed in testFunc callback
        The signature of testFunc should be testFunc(intent, actual, *args, **kwargs)
        """
        if testFuncArgs is None: testFuncArgs = []
        if testFuncKwargs is None: testFuncKwargs = {}
        msgFormatted = ''

        # resolve error message formatting options
        if errorFmt is None:
            # if fmt is set to None it will be picked automatically based on dtype downstream.
            fmt = None
        elif errorFmt.lower()=='auto':
            fmt = self.errorFmt
        else:
            fmt = errorFmt
        if truncate is None: truncate = self.truncate
        
        # bundle all of the format kwargs together
        fmtKwargs = dict([
            ('addbrackets', addbrackets),
            ('fmt', fmt),
            ('squeeze', squeeze),
            ('truncate', truncate),
        ])
        
        # avoid overriding downstream defaults by skipping certain kwargs if their value is None
        if delimiter is not None: fmtKwargs['delimiter'] = delimiter
        if newline is not None: fmtKwargs['newline'] = newline

        fmtKwargs.update(kw)

        # make sure we have arrays and not lists, generators, etc
        intent,actual = _coerceArray(intent), _coerceArray(actual)

        if truncate is not None:
            # truncate the arrays as specified
            intent = intent[:truncate]
            actual = actual[:truncate]

        # make sure the array shapes are the same
        shapeDiff = np.equal(intent.shape, actual.shape)
        shapeBool = np.all(shapeDiff)

        if not shapeBool:
            # string formatting large arrays is surprisingly expensive, only bother doing it if the shapeBool is False
            intentFormatted = nppretty.formatArrayAsArray(intent, **fmtKwargs)
            actualFormatted = nppretty.formatArrayAsArray(actual, **fmtKwargs)

            # print a shape specific message
            msgShape = 'intent array shape and actual array shape do not match'
            msgFormatted = '%s\n' 'intent.size: %d actual.size: %d\n' 'intent.shape: %s\n' 'actual.shape: %s\n' 'shape diff  : %s\n' 'intent: %s\n' 'actual: %s\n' % (msgShape, intent.size, actual.size, intent.shape, actual.shape, shapeDiff, intentFormatted, actualFormatted)

        # do the assert for the shape
        self.assertTrue(shapeBool, msg=msgFormatted)

        # run the test function, accounting for the fact that the test function itself may fail
        try:
            testDiff = testFunc(intent, actual, *testFuncArgs, **testFuncKwargs)
            testBool = np.all(testDiff)
        except (AttributeError,ValueError,TypeError):
            # one array empty: TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
            testDiff = None
            testBool = False

        if not testBool:
            # string formatting large arrays is surprisingly expensive, only bother doing it if the testBool is False
            intentFormatted = nppretty.formatArrayAsArray(intent, **fmtKwargs)
            actualFormatted = nppretty.formatArrayAsArray(actual, **fmtKwargs)

            if testDiff is not None:
                np.set_printoptions(threshold=int(1e5), linewidth=int(1e5))
                testDiffFormatted = testDiff.__str__()
                np.set_printoptions(linewidth=75, threshold=int(1e3))
            else:
                testDiffFormatted = testDiff

            # add some data to the message
            msgFormatted = '%s\n' 'intent.size: %d actual.size: %d\n' 'intent.shape: %s\n' 'actual.shape: %s\n' 'intent: %s\n' 'actual: %s\n' 'diff: %s\n' % (msg, intent.size, actual.size, intent.shape, actual.shape, intentFormatted, actualFormatted, testDiffFormatted)

        # do the assert for the test
        if not testBool: self.fail(msg=msgFormatted)

    def assertArrayAlmostEqual(self, intent, actual, addbrackets=True, delimiter=None, errorFmt=None, msg='arrays not almost equal',
                               newline=None, squeeze=True, truncate=None, rtol=None, atol=None, equal_nan=None, **kw):
        """Assert that all elements of two numpy arrays (or objs that can be coerced into numpy arrays) are equal to within a certain tolerance

        Takes the same optional kwargs as numpy.allclose
        """
        # set up kwargs for np.allclose
        iscloseKwargs = {key:val for key,val in zip(('rtol','atol','equal_nan'), (rtol,atol,equal_nan)) if val is not None}

        # call the function that performs the actual assert
        self._assertArray(intent=intent, actual=actual, addbrackets=addbrackets, delimiter=delimiter,
                          errorFmt=errorFmt, msg=msg, newline=newline, squeeze=squeeze, 
                          truncate=truncate, testFunc=np.isclose, testFuncKwargs=iscloseKwargs, **kw)


    def assertArrayEqual(self, intent, actual, addbrackets=True, delimiter=None, errorFmt=None, msg='arrays not equal',
                         newline=None, squeeze=True, truncate=None, **kw):
        """Assert that all elements of two numpy arrays (or objs that can be coerced into numpy arrays) are equal
        """
        # have to use operator.eq instead of numpy.equal since the later chokes on arrays of strings
        testFunc = operator.eq

        # call the function that performs the actual assert
        self._assertArray(intent=intent, actual=actual, addbrackets=addbrackets, delimiter=delimiter,
                          errorFmt=errorFmt, msg=msg, newline=newline, squeeze=squeeze, 
                          truncate=truncate, testFunc=testFunc, **kw)

    def assertArrayNPEqual(self, intent, actual, addbrackets=True, delimiter=None, errorFmt=None, msg='arrays not equal',
                           newline=None, squeeze=True, truncate=None, **kw):
        """Assert that all elements of two numpy arrays (or objs that can be coerced into numpy arrays) are equal.
        Uses np.testing.assert_equal, which handles nastiness like NaN in a sane fashion.
        """
        # have to use operator.eq instead of numpy.equal since the later chokes on arrays of strings
        testFunc = _npEqual

        # call the function that performs the actual assert
        self._assertArray(intent=intent, actual=actual, addbrackets=addbrackets, delimiter=delimiter,
                          errorFmt=errorFmt, msg=msg, newline=newline, squeeze=squeeze, 
                          truncate=truncate, testFunc=testFunc, **kw)

    # def assertIsInstance(self, obj, cls, *args):
    #     testBool = isinstance(obj, cls)
    #
    #     self.assertTrue(testBool, msg='obj is an instance of class %s, not class %s.' % (type(obj).__name__, cls.__name__)) #msg='obj is an instance of class %s, not class %s.\nobj: %s' % (type(obj).__name__, cls.__name__, obj))

    def assertHasAttr(self, obj, intendedAttr):
        testBool = hasattr(obj, intendedAttr)
        if not testBool: self.fail(msg='obj lacking an attribute. obj: %s, intendedAttr: %s' % (obj, intendedAttr))

    def assertNotHasAttr(self, obj, badAttr):
        testBool = not hasattr(obj, badAttr)

        if not testBool: self.fail(msg='obj has an attribute that it should not have. obj: %s, badAttr: %s' % (obj, badAttr))

    def assertItersEqual(self, itIntended, itActual, msg='Elements of iterators not all equal.', testFunc=None, testArgs=None, testKwargs=None):
        self.assertSequencesEqual(
            seqIntended=[x for x in itIntended],
            seqActual=[x for x in itActual],
            msg=msg,
            testFunc=testFunc,
            testArgs=testArgs,
            testKwargs=testKwargs
        )

    def assertLength(self, lenIntent, actual, msg='length of obj not equal to intent length.'):
        objSize = lambda obj: obj.size if isinstance(obj, np.ndarray) else len(list(obj))

        lenActual = objSize(actual)
        testBool = (lenIntent==lenActual)

        msgFormatted = '\n'.join([
            '%s' % msg,
            'intended len: %d, actual len: %d' % (lenIntent, lenActual),
            'actual: %s' % actual,
            ])

        if not testBool: self.fail(msg=msgFormatted)

    def assertLengthEqual(self, intent, actual, msg='lengths not equal.'):
        objSize = lambda obj: obj.size if isinstance(obj, np.ndarray) else len(list(obj))

        lenIntent = objSize(intent)
        lenActual = objSize(actual)
        testBool = (lenIntent==lenActual)

        msgFormatted = '\n'.join([
            '%s' % msg,
            'intended len: %d, actual len: %d' % (lenIntent, lenActual),
            'intent: %s' % intent,
            'actual: %s' % actual,
            ])

        if not testBool: self.fail(msg=msgFormatted)

    def assertNotRaises(self, expected_exception, *args, **kwargs):
        context = _AssertNotRaisesContext(expected_exception, self)
        try:
            return context.handle('assertNotRaises', args, kwargs)
        finally:
            # bpo-23890: manually break a reference cycle
            context = None

    def assertSequencesEqual(self, seqIntended, seqActual, msg='Elements of sequences not all equal.', truncate=None, testFunc=None, testArgs=None, testKwargs=None):
        if testFunc is None: testFunc = _allWrap(operator.eq)
        if testArgs is None: testArgs = ()
        if testKwargs is None: testKwargs = {}
        msgFormatted = ''

        # coerce generators into proper sequences
        seqIntended = [x for x in seqIntended]
        seqActual = [x for x in seqActual]

        # truncate, if requested
        if truncate is not None:
            # be paranoid about side effects and don't slice at all if None
            seqIntended = seqIntended[:truncate]
            seqActual = seqActual[:truncate]

        lenIntended,lenActual = len(seqIntended), len(seqActual)
        lenMin,lenMax = min(lenIntended, lenActual), max(lenIntended, lenActual)

        testDiff = []
        for elemIntended,elemActual in zip(seqIntended, seqActual):
            try:
                testDiff.append(testFunc(elemIntended, elemActual, *testArgs, **testKwargs))
            except (AttributeError,ValueError):
                testDiff.append(False)


        # if intended and actual have different lens, pad out testDiff with Nones
        lenDiff = lenMax - lenMin
        if lenDiff:
            testDiff.extend([None]*(lenDiff))
            msg = 'Sequences have different numbers of elements.'
            testBool = False
        else:
            # check if all of the elem-by-elem tests passed
            testBool = np.all(testDiff)

        if not testBool:
            msgFormatted = '\n'.join([
                '%s' % msg,
                'intended len: %d actual len: %d' % (lenIntended, lenActual),
                'intent: %s' % seqIntended,
                'actual: %s' % seqActual,
                'diff: %s' % testDiff,
            ])

        if not testBool: self.fail(msg=msgFormatted)

    def assertSetsEqual(self, set1, set2):
        testBool = set1==set2
        if not testBool: self.fail(msg='sets not equal: %s\n%s' % (set1, set2))

    def assertShape(self, shapeIntent, actual, msg='shape of actual not equal to intent shape.'):
        shapeActual = np.asarray(actual).shape
        testBool = (shapeIntent==shapeActual)

        msgFormatted = '\n'.join([
            '%s' % msg,
            'intended shape: %s, actual shape: %s' % (shapeIntent, shapeActual),
            'actual: %s' % actual,
            ])

        if not testBool: self.fail(msg=msgFormatted)

    def assertShortArraysEqual(self, arr1, arr2):
        long,short = (arr1,arr2) if len(arr1) > len(arr2) else (arr2,arr1)
        self.assertArrayEqual(short, long[:len(short)], msg='first %d elements of arrays not equal' % len(short))
