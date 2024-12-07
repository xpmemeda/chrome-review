import smartptr

shared_obj = smartptr.SharedObject()
shared_obj.show_this()
smartptr.SharedObject.accept_object(shared_obj)  # copy.
smartptr.SharedObject.accept_ref(shared_obj)  # no copy.
smartptr.SharedObject.accept_raw_ptr(shared_obj)  # no copy.
smartptr.SharedObject.accept_shared_ptr(shared_obj)  # no copy.

unique_obj = smartptr.UniqueObject()
unique_obj.show_this()
smartptr.UniqueObject.accept_object(unique_obj)  # copy.
smartptr.UniqueObject.accept_ref(unique_obj)  # no copy.
smartptr.UniqueObject.accept_raw_ptr(unique_obj)  # no copy.
