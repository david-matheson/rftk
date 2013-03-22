###############################################################################
# Warning: This file modifies the python interface for buffer collections
###############################################################################

import itertools
import buffers as buffers

def _buffer_add(self, name, b):
    if not buffers.is_buffer(b):
        b = buffers.as_buffer(b)
    class_name = b.__class__.__name__
    function_name = '%s%s' % ('Add', class_name)
    if hasattr(self, function_name):
        function = getattr(self, function_name)
        return function(name, b)
    else:
        raise Exception('BufferCollection.Add failed because %s does not exist' % function_name)

BufferCollection.AddBuffer = _buffer_add


def _get_buffer(self, name):
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

BufferCollection.GetBuffer = _get_buffer


buffers._BufferCollection = buffers.BufferCollection

def _create_buffer(**kwargs):
    collection = buffers._BufferCollection()
    for key, value in kwargs.iteritems():
        collection.AddBuffer(key, value)
    return collection

buffers.BufferCollection = _create_buffer