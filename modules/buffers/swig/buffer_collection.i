%module buffers
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferCollection.h"
%}


%pythoncode %{
import itertools
import converters as converters
%}

%extend BufferCollection {
%insert("python") %{

    def __init__(self, *args, **kwargs):
        this = _buffers.new_BufferCollection(*args)
        try: self.this.append(this)
        except: self.this = this
        for key, value in kwargs.iteritems():
            self.AddBuffer(key, value)

    def AddBuffer(self, name, b):
        if not converters.is_buffer(b):
            b = converters.as_buffer(b)
        class_name = b.__class__.__name__
        function_name = '%s%s' % ('Add', class_name)
        if hasattr(self, function_name):
            function = getattr(self, function_name)
            return function(name, b)
        else:
            raise Exception('BufferCollection.Add failed because %s does not exist' % function_name)

    def GetBuffer(self, name):
        data_type_list = ['Float32', 'Float64', 'Int32', 'Int64']
        container_type_list = ['Vector', 'Matrix', 'SparseMatrix', 'Tensor3']

        for (data_type, container_type) in itertools.product(data_type_list, container_type_list):
            buffer_type_name = "%s%sBuffer" % (data_type, container_type)
            has_function_name = "Has%s" % buffer_type_name
            has_function = getattr(self, has_function_name)
            if has_function(name):
                get_function_name = "Get%s" % buffer_type_name
                get_function = getattr(self, get_function_name)
                return get_function(name)
        raise Exception('BufferCollection.GetBuffer failed because %s does not exist' % name)

    def __getstate__(self):
        keys = self.GetKeys()
        data_dict = {}
        for k in keys:
            data_dict[k] = self.GetBuffer(k) 
        return data_dict

    def __setstate__(self,data_dict):
        self.__init__()
        for key, value in data_dict.iteritems():
            self.AddBuffer(key, value)

%}
}
