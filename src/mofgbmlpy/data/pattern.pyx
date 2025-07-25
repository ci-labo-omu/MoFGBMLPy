import copy
import numpy as np
cimport numpy as cnp
import cython
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic, ClassLabelBasicCpp
from mofgbmlpy.data.pattern cimport Pattern, PatternCpp
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.data.class_label.class_label_multi cimport ClassLabelMultiCpp, ClassLabelMulti
from mofgbmlpy.data.class_label.class_label_wrapper cimport wrap_class_label
from libcpp.vector cimport vector

cdef class Pattern:
    """Pattern (row) of a dataset. Contains a vector of attributes and a class label

    Attributes:
        __id (int): ID of the pattern
        __attributes_vector (double[]): Array of the attributes. The size of this array is the number of dimensions
        __target_class (AbstractClassLabel): Class label associated to this pattern
    """

    def __cinit__(self, int pattern_id, double[:] attributes_vector=None, AbstractClassLabel target_class=None, do_init=True):
        """Constructor

        Args:
            pattern_id (int): ID of the pattern
            attributes_vector (double[]): Array of the attributes. The size of this array is the number of dimensions
            target_class (AbstractClassLabel): Class label associated to this pattern
            do_init (bool): If True, the object is initialized, otherwise it is not
        """
        if not do_init:
            self.ptr = NULL
            return

        cdef vector[double] cpp_attributes_vector
        if attributes_vector is not None and attributes_vector.shape[0] != 0:
            cpp_attributes_vector.reserve(attributes_vector.shape[0])
            for i in range(attributes_vector.shape[0]):
                cpp_attributes_vector.push_back(attributes_vector[i])

        cdef AbstractClassLabelCpp * class_ptr = NULL
        if target_class is not None:
            class_ptr = target_class.get_ptr().clone()

        self.ptr = new PatternCpp(pattern_id, cpp_attributes_vector, class_ptr)

    def __dealloc__(self):
        """Destructor"""
        if self.ptr != NULL:
            del self.ptr

    cpdef int get_id(self):
        """Get the ID
        
        Returns:
            int: ID of the pattern
        """
        return self.ptr.get_id()

    cpdef double[:] get_attributes_vector(self):
        """Get the attributes vector
        
        Returns:
            double[]: Array of attributes values
        """
        return np.array(self.ptr.get_attributes_vector(), np.float64)

    cpdef double get_attribute_value(self, int index):
        """Get the attribute value at the given index
        
        Args:
            index (int): Index of the attribute whose value is returned 

        Returns:
            double: Attribute value
        """
        return self.ptr.get_attribute_value(index)

    cpdef object get_target_class(self):
        """Get the target class label of this pattern
        
        Returns:
            object: Target class label. Either a int or an array of int (multi label)
        """
        cdef AbstractClassLabelCpp* target_class_ptr = self.ptr.get_target_class()
        return wrap_class_label(target_class_ptr)

    cpdef int get_num_dim(self):
        """Get the number of dimensions of the attribute vector
        
        Returns:
            int: Number of dimensions
        """
        return self.ptr.get_num_dim()

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            str: String representation
        """
        return self.ptr.to_string().decode('utf-8')

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = Pattern.wrap(self.ptr)
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, Pattern):
            return False
        cdef Pattern other_c = <Pattern> other
        return other_c.ptr[0] == self.ptr[0]

    cpdef void set_attribute_value(self, int index, float new_value):
        self.ptr.set_attribute_value(index, new_value)

    @staticmethod
    cdef Pattern wrap(PatternCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")
        cdef Pattern new_object = Pattern(0, do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object
