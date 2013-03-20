%pythoncode %{
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

BufferCollection.Add = _buffer_add

_buffer_type_names = ['Float32VectorBuffer', 
                'Float64VectorBuffer', 
                'Int32VectorBuffer', 
                'Int64VectorBuffer',
                'Float32MatrixBuffer', 
                'Float64MatrixBuffer', 
                'Int32MatrixBuffer', 
                'Int64MatrixBuffer',
                'Float32Tensor3Buffer', 
                'Float64Tensor3Buffer', 
                'Int32VectorTensor3Buffer', 
                'Int64VectorTensor3Buffer',
                'Float32SparseMatrixBuffer', 
                'Float64SparseMatrixBuffer', 
                'Int32VectorSparseMatrixBuffer', 
                'Int64VectorSparseMatrix']

def _get_buffer(self, name):
    for buffer_type_name in _buffer_type_names:
        has_function_name = "%s%s" % ('Has', buffer_type_name)
        has_function = getattr(self, has_function_name)
        if has_function(name):
            get_function_name = "%s%s" % ('Get', buffer_type_name)
            get_function = getattr(self, get_function_name)
            return get_function(name)
    raise Exception('BufferCollection.GetBuffer failed because %s does not exist' % name)

BufferCollection.GetBuffer = _get_buffer

buffers._BufferCollection = buffers.BufferCollection

def _create_buffer(**kwargs):
    collection = buffers._BufferCollection()
    for key, value in kwargs.iteritems():
        collection.Add(key, value)
    return collection

buffers.BufferCollection = _create_buffer

%}