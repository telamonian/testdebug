# PYTHON_ARGCOMPLETE_OK
import argparse
from argparse import ArgumentParser, SUPPRESS
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
import re
import sys

try:
    import argcomplete
    from argcomplete.completers import ChoicesCompleter
    doComplete = True
except ImportError:
    doComplete = False

__all__ = ['ArgProperty', 'ArgGroup', 'ConversionFuncs', 'PostProcessFuncs', 'Parser']

def ChainFilter(*elems, **kwargs):
    pred = kwargs['pred'] if 'pred' in kwargs else None

    return chain.from_iterable(filter(pred, elems))

def IsContainer(x):
    '''
    tests if x is an instance of one of the builtin container types
    '''
    return isinstance(x, (dict, frozenset, list, set, tuple))

listSplitRe = re.compile(r'[\s,;:]')
def ListSplit(s):
    """Splits a string into a list of values. Valid delimiters are ,:; and any space character
    """
    return listSplitRe.split(s)

def Setify(x):
    '''
    if x is an instance of a builtin container, convert it to a set. Otherwise, place x into a set
    '''
    if IsContainer(x):
        return set(x)
    else:
        return {x}

class ArgProperty(object):
    '''
    object representing single arg to be passed to ArgumentParser.add_argument()
    '''
    # kwargs (and default values) for extra functionality that ArgProperty understands but ArgumentParser.add_argument() does not
    extraKwargs = [('completeChoices', None), ('propDict', None), ('suppress', False), ('trueDefault', None)]

    # property that generates the *args that get passed to .add_argument
    @property
    def argsForAdd(self):
        return self.flags

    @property
    def flag(self):
        return self.flags[0]

    # property that generates the **kwargs that get passed to .add_argument
    @property
    def kwargsForAdd(self):
        return self.kwargs

    def __init__(self, *flags, **kwargs):
        # init attrs from args
        self.flags = flags

        # pop any items from kwargs that shouldn't get passed to ArgumentParser.add_argument() and handle them
        self.initAttrsFromKwargs(kwargs)

        # figure out the arg name/dest
        self.name = self.genName()

        # other initialization
        self.initDefault()
        self.initType()

    def initAttrsFromKwargs(self, kwargs):
        # handle the 'trueDefault' kwarg, if present. This can be set if we need to tell the difference between the parser receiving no argVal and the user entering a value==the default value
        for extraKwarg in self.extraKwargs:
            if extraKwarg[0] in kwargs:
                self.__setattr__(extraKwarg[0], kwargs.pop(extraKwarg[0]))
            else:
                self.__setattr__(*extraKwarg)

        self.kwargs = kwargs
        if self.propDict is not None:
            self.kwargs.update(self.propDict)

    def initDefault(self):
        if 'default' not in self.kwargs:
            if self.suppress:
                self.kwargs['default'] = argparse.SUPPRESS
        elif self.kwargs['default']==argparse.SUPPRESS:
            self.suppress = True

    def initPostProcess(self, tipe):
        # add a post processing func if required, as determined by tipe
        if tipe in PostProcessFuncs.__dict__:
            self.postProcess = PostProcessFuncs.__dict__[tipe].__func__

    def initType(self):
        # if 'type' in kwargs, see if we need to replace a string with a conversion function
        if 'type' in self.kwargs and isinstance(self.kwargs['type'], str) and hasattr(ConversionFuncs, self.kwargs['type']):
            tipe = self.kwargs['type']
            self.kwargs['type'] = ConversionFuncs.__dict__[tipe].__func__
            self.initPostProcess(tipe)

    def genName(self):
        """Gets the argProperty name, following the conventions laid out in the argparse docs, section `16.4.3.11. dest`
        """
        # first, check for an explicitly set dest
        if 'dest' in self.kwargs:
            return self.kwargs['dest']

        # next, check for if the first flag does not have the '-' dinks, as for a positional arg
        if not self.flags[0].startswith('-'):
            return self.flags[0]

        # next, look for the first flag that starts with the double '--', indicative of a long form optional arg name
        for flag in self.flags:
            if flag.startswith('--'):
                # strip the dinks before returning
                return flag.lstrip('-')

        # next, look for the first flag that starts with '-', indicative of a short form optional arg name
        for flag in self.flags:
            if flag.startswith('-'):
                # strip the dinks before returning
                return flag.lstrip('-')

        # if we got to this point and nothing has been found, raise an error
        raise ValueError('During ArgProperty initialization, no valid name was found in self.flags or self.kwargs. self.flags: %s, self.kwargs: %s' % (self.flags, self.kwargs))

class ArgGroup(object):
    '''
    object representing group of args to be passed to ArgumentParser.add_argument
    '''
    # property that generates the *args that get passed to .add_argument
    @property
    def argsForAdd(self):
        return [argProp.argsForAdd for argProp in self.values()]

    # property that generates the **kwargs that get passed to .add_argument
    @property
    def kwargsForAdd(self):
        return [argProp.kwargsForAdd for argProp in self.values()]

    @property
    def argNames(self):
        return [argProp.name for argProp in self.values()]

    # short aliases
    @property
    def argProps(self):
        return self.argProperties

    def __init__(self, name, *argProperties):
        # init attrs from args
        self.name = name
        self.argProperties = OrderedDict()

        self.addArgProperty(*argProperties)

    def __contains__(self, key):
        return key in self.argNames

    def __iter__(self):
        return self.argProperties.__iter__()

    # other iterators
    def keys(self):
        return self.argProperties.keys()

    def items(self):
        return self.argProperties.items()

    def values(self):
        return self.argProperties.values()

    def addArgProperty(self, *argProperties):
        for argProp in argProperties:
            self.argProperties[argProp.name] = argProp

    def pop(self, name):
        return self.argProperties.pop(name)

    def update(self, other):
        self.argProperties.update(other.argProperties)

class Parser(object):
    cmdLineExtra = None

    @property
    def kwargs(self):
        return self.args.__dict__

    def __init__(self, argGroups=None, description='', **kwargs):
        # self.args is an argparse.Namespace object that gets set the first time this Parser instance is run
        self.args = None
        self.ran = False

        # init attrs from *argProperties
        self.argGroups = OrderedDict()
        self.argProperties = OrderedDict()
        self.defaultOverrideDict = kwargs
        self.description = description

        self.addArgGroup(*ChainFilter(argGroups, self.genDefaultArgGroups()))

        # this will be called again as the first line of .run()
        self.initParser()

    def __contains__(self, key):
        return key in self.kwargs

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def initParser(self):
        # turns out this works best if it isn't called until .run() is called
        self.parser = ArgumentParser(self.description)
        for argProp in (argProp for argGroup in self.argGroups.values() for argProp in argGroup.values()):
            if argProp.name in self.defaultOverrideDict:
                argProp.kwargs['default'] = self.defaultOverrideDict[argProp.name]

            if argProp.completeChoices is not None and doComplete:
                self.parser.add_argument(*argProp.argsForAdd, **argProp.kwargsForAdd).completer=ChoicesCompleter(argProp.completeChoices)
            else:
                self.parser.add_argument(*argProp.argsForAdd, **argProp.kwargsForAdd)

    def addArgGroup(self, *argGroups):
        for argGroup in argGroups:
            if argGroup.name in self.argGroups:
                # pop all of the old arg props
                for argProp in argGroup.values():
                    self.popArgProperty(argProp.name)

                # merge the new and old argGroups
                self.argGroups[argGroup.name].update(argGroup)
            else:
                # we can skip the cleanup in this case
                self.argGroups[argGroup.name] = argGroup

            for argProp in self.argGroups[argGroup.name].values():
                self.argProperties[argProp.name] = argProp

    def addArgProperty(self, argProp, group):
        self.argProperties[argProp.name] = argProp
        if group is not None:
            self.argGroups[group].addArgProperty(argProp)

    def genArgGroupSet(self, group=None):
        if group is not None:
            return set(self.argGroups[g] for g in Setify(group))
        else:
            return set(self.argGroups.values())

    def genDefaultArgGroups(self):
        # hook for subclasses that want to define a default set of argGroups
        return None

    def get(self, name):
        """Getter for parsed arg values (need to use .run() first)

        :param name: arg name
        :return: arg value
        """
        try:
            return self.args.__getattribute__(name)
        except AttributeError:
            # fix the whole error arising from the fact that ArgumentParser switches '_' for '-' in argument names
            return self.args.__getattribute__('_'.join(name.split('-')))

    def getArgDict(self, group=None, exclude=None, subs=None):
        argGroupSet = self.genArgGroupSet(group)
        return OrderedDict(((argName,argVal) for argName,argVal in self.getArgTupsFromArgGroupSet(argGroupSet, exclude, subs)))

    def getArgProperty(self, name):
        try:
            return self.argProperties[name]
        except KeyError:
            return self.argProperties['-'.join(name.split('_'))]

    def getArgTupsFromArgGroupSet(self, argGroupSet, cmdLine=False, exclude=None, subs=None):
        exclude = exclude if exclude is not None else []
        subs = subs if subs is not None else []

        argTups = []
        for argName,argProp in [argPropTup for argGroup in argGroupSet for argPropTup in argGroup.items() if argPropTup[0] not in exclude]: # and not argPropTup[1].suppress]:
            # dest may not match argName, so make sure we're using the right argKey
            argKey = argProp.kwargs['dest'] if 'dest' in argProp.kwargs else argName

            try:
                argVal = self.get(argKey)
            except AttributeError:
                # there was no user entry corresponding to this argProp and argProp.default==argparse.SUPPRESS
                continue

            # replace values of args as specified by the subs param
            for sub in subs:
                if argName==sub[0] or argName=='-'.join(sub[0].split('_')):
                    argVal = sub[1]

            # workaround to help tell the difference between no user input and user input==default value
            if argVal==self.getDefault(argName) and argProp.trueDefault is not None:
                argVal = argProp.trueDefault

            retArgName = argProp.flag if cmdLine else argKey
            argTups.append((retArgName, argVal))
        return argTups

    def getCmdLine(self, group=None, exclude=None, subs=None, suppressBool=True):
        argGroupSet = self.genArgGroupSet(group)

        cmdLineTups = []
        for argName,argVal in self.getArgTupsFromArgGroupSet(argGroupSet=argGroupSet, cmdLine=True, exclude=exclude, subs=subs):
            if isinstance(argVal, (tuple, list, set, frozenset)):
                cmdLineTups.append((argName, ','.join(argVal)))
            else:
                cmdLineTups.append((argName, str(argVal)))

        if suppressBool:
            return [argToken for cmdLineTup in cmdLineTups for argToken in cmdLineTup if (argToken!='False' and argToken!='True')]
        else:
            return [argToken for cmdLineTup in cmdLineTups for argToken in cmdLineTup]

    def getDefault(self, name):
        return self.getArgProperty(name).kwargs.get('default', None)

    def hasArgProp(self, name):
        return self.resolveArgName(name) is not None

    def pop(self, name):
        """Pops a parsed arg val from self.args.__dict__

        :param name: arg name
        :return: popped arg val
        """
        return self.kwargs.pop(name)

    def popArgProperty(self, name):
        for argGroup in self.argGroups.values():
            try:
                argGroup.pop(name)
            except KeyError:
                pass
        return self.argProperties.pop(name)

    def postProcess(self):
        for argName,argProp in self.argProperties.items():
            if hasattr(argProp, 'postProcess'):
                try:
                    self.set(argName, argProp.postProcess(self.get(argName)))
                except AttributeError:
                    pass

    def resolveArgName(self, name):
        """- Serves two purposes
            - checks if the passed name corresponds to a known command line arg
            - if the check passes, it returns the version of the name with all of the '-' replaced with '_'
        """
        if name in self.argProperties:
            # name was known, but it may contain '-', so fix that and return
            return '_'.join(name.split('-'))
        else:
            # internally stored attr names can't have '-' in them, but cmd line args can, so check if reversing their replacement helps
            emDashName = '-'.join(name.split('_'))
            if emDashName in self.argProperties:
                # name was known, but it may contain '-', so fix that and return
                return '_'.join(name.split('-'))
            else:
                # the name was not recognized, raise a KeyError
                raise KeyError('name does not correspond to any known args. name: %s, self.argProperties.keys(): %s' % (name, list(self.argProperties.keys())))

    def run(self, cmdLine=None, cmdLineExtra=None, rerun=False):
        """if you want to parse a list of args aside from sys.argv, specify it in cmdLine
           if you want to parse a list of args in addition to sys.argv, specify it in cmdLineExtra

           self.__class__.cmdLineExtra is prepended to cmdLineExtra, if it exists
        """
        # make the run method idempotent by skipping it if this Parser instance has already .run(). Will rerun anyway if requested
        if self.ran and not rerun:
            return

        self._run(cmdLine=cmdLine, cmdLineExtra=cmdLineExtra)

    def _run(self, cmdLine, cmdLineExtra):
        """A hook for subclasses to override .run() without screwing with the idempotency stuff
        """
        self.initParser()

        if doComplete:
            argcomplete.autocomplete(self.parser)

        cmdLineToks = sys.argv[1:] if cmdLine is None else cmdLine.split()
        cmdLineToks.extend([extraTok for extra in (self.cmdLineExtra, cmdLineExtra) if extra is not None for extraTok in extra.split()])

        self.args = self.parser.parse_args(cmdLineToks)

        self.postProcess()
        self.ran = True

    def set(self, name, val):
        """Setter for parsed arg vals

        :param name: arg name
        :param val: arg val
        :return: None
        """
        # fix the whole error arising from the fact that ArgumentParser switches '_' for '-' in argument names
        self.args.__setattr__(self.resolveArgName(name), val)

    def update(self, *args, **kwargs):
        [kwargs.update(argDict) for argDict in args]

        for name,val in kwargs.items():
            self.set(name=name, val=val)


class ConversionFuncs(object):
    '''
    type conversion functions for the 'type' kwarg
    '''
    @staticmethod
    def delimiterSubClosure(findDelim, replDelim):
        '''
        conversion for option values that need a single character substitution
        for example 'protein_and_not_name_H' -> 'protein and not name H'
        '''
        def delimiterSub(s):
            return replDelim.join(s.split(findDelim))
        return delimiterSub

    @staticmethod
    def intRange(s):
        '''
        conversion for option values containing ranges of integers specified as a-b,c-d,...
        for example 2,3-5,19,27,41-43 -> [2,3,4,5,19,27,41,42,43]
        '''
        vals = []
        for subS in ListSplit(s):
            if '-' in subS:
                # the map serves to convert everything in the list returned by .split to integers
                vals+=ConversionFuncs.rangeToEnd(*map(int, subS.split('-')))
            else:
                vals.append(int(subS))
        return vals

    @staticmethod
    def rangeToEnd(start, stop):
        '''
        version of built-in range() that includes the stop val in the return
        '''
        return range(start, stop+1)

    @staticmethod
    def strList(s):
        '''
        conversion for option values containing multiple strings specified as str0,str1...
        for example foo,bar -> ['foo', 'bar']
        '''
        return ListSplit(s)

    @staticmethod
    def varEqualsVal(s):
        '''
        conversion for option values containing variable names and values specified as name0=val0,name1=val1...
        for example apple=red,foo=bar -> {'apple': 'red', 'foo': 'bar'}
        '''
        varDict = {}
        for subS in ListSplit(s):
            if '=' in subS:
                key,val = subS.split('=')
                varDict[key] = val
            else:
                raise ValueError('malformed variable specification in option value. should be in the form name0=val0,name1=val1...\n option value: %s' % s)
        return varDict

class PostProcessFuncs(object):
    """
    functions that can be called on argVals after initial parsing has completed. Usually for assisting with combinations of nargs='+' and a ConversionFunc
    """
    @staticmethod
    def intRange(lol):
        """
        if type=intRange and nargs='*' are used together, we get back a mixed list of ints and list-of-ints, so this'll flatten it
        """
        flat = []
        if lol is not None:
            # it might be a single int, or it might be a list, so we'll just call it a chunk
            for chunk in lol:
                if isinstance(chunk, int):
                    flat.append(chunk)
                else:
                    flat.extend(chunk)

        return flat

        # flatIR = []
        # if iR is not None:
        #     # it might be a single int, or it might be a list, so we'll just call it a chunk
        #     for intChunk in iR:
        #         try:
        #             for i in intChunk:
        #                 flatIR.append(i)
        #         except TypeError:
        #             flatIR.append(intChunk)
        #
        # return flatIR

    @staticmethod
    def strList(lol):
        """
        if type=strList and nargs='*' are used together, we get back a mixed list of val and list-of-vals, so this'll flatten it
        
        :lol: list of lists
        """
        flat = []
        if lol is not None:
            # it might be a single str, or it might be a list, so we'll just call it a chunk
            for chunk in lol:
                if isinstance(chunk, str):
                    flat.append(chunk)
                else:
                    flat.extend(chunk)

        return flat
