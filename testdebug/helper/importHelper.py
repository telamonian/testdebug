from imp import find_module
import importlib
from pathlib import Path
import pkgutil
from pkgutil import iter_modules
import re

__all__ = ['DeepImport', 'FromImport', 'ShallowImport',
           'GetPathFromModuleName', 'GetSubmoduleNamesFromPackageName',
           'ImportFromFullname', 'NormalizeScriptExt']

def FromImport(modname, *names):
    """Dynamic version of the `from mod import ...` syntax.
    Equivalent to `from modname import name[0],name[1],...`, except that the
    imported objects are returned as a list instead of being assigned to
    variables matching the names.
    """
    mod = __import__(modname, globals(), locals(), names)
    return [getattr(mod, name) for name in names]

def _filterCheck(s, regexFilter, clusivity):
    if clusivity=='exclude':
        # if s matches a specified filter, skip this
        return bool(regexFilter.search(s))
    elif clusivity=='include':
        # if s does not match a specified filter, skip this
        return not regexFilter.search(s)
    else:
        raise ValueError("Invalid clusivity.\n"
                         "clusivity: %s" % clusivity)

def ShallowImport(path=None, name=None, fullname=None, attrs=None, locals=None,
                  attrFilter=None, attrFilterClusivity='include',
                  nameFilter=None, nameFilterClusivity='include',
                  skipModules=False, skipPkgs=False, _deep=False, onerror=None):
    """imports every module and package in path into the name namespace
    eg. ShallowImport(path='/real/system/path', name='foo.bar') would
    return {'foo.bar.re': <foo.bar.re mod>, 'foo.bar.baa': <foo.bar.baa mod>}

    if fullname (ie a dotted python path) is used instead of path and name,
    the path and name of the pkg containing fullname are used

    use in a __init__.py file verbatim like this:
        modDict = ShallowImport(path=__path__, name=__name__)
        locals().update(modDict)
    """
    if path is None == fullname is None:
        raise ValueError("Exactly one of (path, fullname) should be set.\n"
                         "path: %s, name: %s" % (path, name))

    if attrFilter is not None: attrFilter = re.compile(attrFilter)
    if nameFilter is not None: nameFilter = re.compile(nameFilter)
    iterFunc = pkgutil.walk_packages if _deep else pkgutil.iter_modules

    if path is None:
        try:
            importerContaining = next(pkgutil.iter_importers(fullname=fullname))
            path = [importerContaining.path]    # path needs to be a list
            name = '.'.join(fullname.split('.')[:-1])
        except AttributeError:
            raise ValueError("Could not determine path and name from fullname.\n"
                             "fullname: %s" % fullname)
    else:
        if isinstance(path, (str, Path)):
            # the pkg iter funcs expect path to be a list
            path = [str(path)]
        else:
            # ensure the contents of path are str
            path = [str(pth) for pth in path]

            # TODO: add support for multiple pth in path
            if len(path) != 1:
                raise ValueError("path should contain exactly one pth.\n"
                                 "path: %s" % path)

    # wihtout a prefix, everything will be imported directly in the caller's namespace
    prefix = name + '.' if name else ''

    importedDict = {}
    for importer, modName, isPkg in iterFunc(path=path, prefix=prefix, onerror=onerror):
        if modName.split('.')[-1][:4]=='old_':
            # this is code marked as disabled/deprecated, skip it
            continue
        if skipPkgs and isPkg:
            # we've been told to skip packages and this is a package
            continue
        if skipModules and not isPkg:
            # we've been told to skip modules and this is a module
            continue
        if nameFilter is not None and _filterCheck(modName, nameFilter, nameFilterClusivity):
            continue

        mod = importer.find_module(modName).load_module(modName)

        # if shortNames:
        #     # strip any parent packages off of the mod's dot-name
        #     modName = modName.split(name+'.')[-1]

        if attrs is not None:
            # iter through the provided attrs
            for attr in attrs:
                if attr=='*':
                    # import all attrs listed in module's __all__
                    for allattr in getattr(mod, '__all__', []):
                        if attrFilter is not None and _filterCheck(allattr, attrFilter, attrFilterClusivity):
                            continue

                        importedDict[allattr] = getattr(mod, allattr)
                elif attr=='.':
                    # import the top level module
                    importedDict[modName] = mod
                else:
                    # import the specific attr, if not filtered
                    if attrFilter is not None and _filterCheck(attr, attrFilter, attrFilterClusivity):
                        continue

                    importedDict[attr] = getattr(mod, attr)
        else:
            # just import the top level module
            importedDict[modName] = mod

    if locals is not None:
        # hack the dynamic imports directly into the supplied locals dict
        locals.update(importedDict)

    return importedDict


def DeepImport(path=None, name=None, fullname=None, attrs=None, locals=None,
               attrFilter=None, attrFilterClusivity='include',
               nameFilter=None, nameFilterClusivity='include',
               skipModules=False, skipPkgs=False, onerror=None):
    """Same as ShallowImport except that it recursively walks the submodules
    """
    return ShallowImport(
        path=path,
        name=name,
        fullname=fullname,
        attrs=attrs,
        locals=locals,
        attrFilter=attrFilter,
        attrFilterClusivity=attrFilterClusivity,
        nameFilter=nameFilter,
        nameFilterClusivity=nameFilterClusivity,
        skipModules=skipModules,
        skipPkgs=skipPkgs,
        _deep=True,
        onerror=onerror)

def GetPathFromModuleName(name, normalizeScriptExt=True):
    nameParts = name.split('.')
    containingName, name = '.'.join(nameParts[:-1]), nameParts[-1]
    # careful! trying to import a pyspark script directly will likely cause an exception at the 'import pyspark' line, so we instead import only the module containing the script
    containingModule = importlib.import_module(containingName)
    fmOut = find_module(name, containingModule.__path__)
    # find_module opens the script as a text file, so we'll close it
    fmOut[0].close()
    return fmOut[1] if not normalizeScriptExt else NormalizeScriptExt(fmOut[1])

def GetSubmoduleNamesFromPackageName(name):
    # iter_modules() returns an iter of tuples in the form (module_loader, name, ispkg)
    return ['.'.join([name, submodTup[1]]) for submodTup in iter_modules(importlib.import_module(name).__path__)]

def ImportFromFullname(fullname, doRaise=True):
    nameparts = fullname.split('.')
    modname,name = nameparts[:-1],nameparts[-1]

    try:
        return FromImport(modname, name)[0]
    except ImportError as e:
        # we reach here if we couldn't import the obj at fullname
        if doRaise:
            raise e
    return None

def NormalizeScriptExt(scriptPth, ext='.py'):
    # depending on installation details, .__path__ might point to a .pyc file instead of a .py file, so fix that
    return str(Path(scriptPth).with_suffix(ext))
