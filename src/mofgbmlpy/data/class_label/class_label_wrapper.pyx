from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasicCpp, ClassLabelBasic
from mofgbmlpy.data.class_label.class_label_multi cimport ClassLabelMultiCpp, ClassLabelMulti
from libcpp.cast cimport dynamic_cast

from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel, AbstractClassLabelCpp

ctypedef ClassLabelBasicCpp* BasicPtr
ctypedef ClassLabelMultiCpp* MultiPtr

cdef AbstractClassLabel wrap_class_label(AbstractClassLabelCpp * wrapped_ptr):
    cdef BasicPtr basic_ptr = dynamic_cast[BasicPtr](wrapped_ptr)
    if basic_ptr != NULL:
        return ClassLabelBasic.wrap(basic_ptr)

    cdef MultiPtr multi_ptr = dynamic_cast[MultiPtr](wrapped_ptr)
    if multi_ptr != NULL:
        return ClassLabelMulti.wrap(multi_ptr)

    raise TypeError("Unknown class label type")