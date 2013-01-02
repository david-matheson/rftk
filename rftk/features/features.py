# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.40
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.
# This file is compatible with both classic and new-style classes.

from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_features', [dirname(__file__)])
        except ImportError:
            import _features
            return _features
        if fp is not None:
            try:
                _mod = imp.load_module('_features', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _features = swig_import_helper()
    del swig_import_helper
else:
    import _features
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


import assert_util
import buffers
class FeatureExtractorI(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FeatureExtractorI, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FeatureExtractorI, name)
    __repr__ = _swig_repr
    def Extract(self, *args): return _features.FeatureExtractorI_Extract(self, *args)
    def GetUID(self): return _features.FeatureExtractorI_GetUID(self)
    def __init__(self): 
        this = _features.new_FeatureExtractorI()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _features.delete_FeatureExtractorI
    __del__ = lambda self : None;
FeatureExtractorI_swigregister = _features.FeatureExtractorI_swigregister
FeatureExtractorI_swigregister(FeatureExtractorI)

class AxisAlignedFeatureExtractor(FeatureExtractorI):
    __swig_setmethods__ = {}
    for _s in [FeatureExtractorI]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, AxisAlignedFeatureExtractor, name, value)
    __swig_getmethods__ = {}
    for _s in [FeatureExtractorI]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, AxisAlignedFeatureExtractor, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _features.new_AxisAlignedFeatureExtractor(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _features.delete_AxisAlignedFeatureExtractor
    __del__ = lambda self : None;
    def GetUID(self): return _features.AxisAlignedFeatureExtractor_GetUID(self)
    def Extract(self, *args): return _features.AxisAlignedFeatureExtractor_Extract(self, *args)
    __swig_setmethods__["mXs"] = _features.AxisAlignedFeatureExtractor_mXs_set
    __swig_getmethods__["mXs"] = _features.AxisAlignedFeatureExtractor_mXs_get
    if _newclass:mXs = _swig_property(_features.AxisAlignedFeatureExtractor_mXs_get, _features.AxisAlignedFeatureExtractor_mXs_set)
AxisAlignedFeatureExtractor_swigregister = _features.AxisAlignedFeatureExtractor_swigregister
AxisAlignedFeatureExtractor_swigregister(AxisAlignedFeatureExtractor)



