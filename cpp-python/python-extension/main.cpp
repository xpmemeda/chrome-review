#include <Python.h>
#include "structmember.h"

typedef struct {
  PyObject_HEAD
  PyObject *name;
  PyObject *age;
} PyDogObject;

static PyObject *Dog_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyDogObject *self;
  self = (PyDogObject *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->name = PyUnicode_FromString("");
    if (self->name == NULL) {
      Py_DECREF(self);
      return NULL;
    }
    self->age = PyLong_FromLong(0);
    if (self->age == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject *)self;
}

static int Dog_init(PyDogObject *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {"name", "age", NULL};
  PyObject *name = NULL, *age = NULL, *tmp;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &name,
                                   &age))
    return -1;

  if (name) {
    tmp = self->name;
    Py_INCREF(name);
    self->name = name;
    Py_XDECREF(tmp);
  }

  if (age) {
    tmp = self->age;
    Py_INCREF(age);
    self->age = age;
    Py_XDECREF(tmp);
  }
  return 0;
}

static void Dog_dealloc(PyDogObject *self) {
  Py_XDECREF(self->name);
  Py_XDECREF(self->age);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Dog_call(PyDogObject *self, PyObject *args, PyObject *kwds) {
  PyObject *woof = PyUnicode_FromString("woof...woof...");
  return woof;
}

static PyMemberDef Dog_members[] = {
    {"name", T_OBJECT_EX, offsetof(PyDogObject, name), 0, "name"},
    {"age", T_OBJECT_EX, offsetof(PyDogObject, age), 0, "age"},
    {NULL} /* Sentinel */
};

PyTypeObject PyDog_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "Dog",                                /* tp_name */
    sizeof(PyDogObject),                  /* tp_basicsize */
    0,                                    /* tp_itemsize */
    (destructor)Dog_dealloc,              /* tp_dealloc */
    0,                                    /* tp_vectorcall_offset */
    0,                                    /* tp_getattr */
    0,                                    /* tp_setattr */
    0,                                    /* tp_as_async */
    0,                                    /* tp_repr */
    0,                                    /* tp_as_number */
    0,                                    /* tp_as_sequence */
    0,                                    /* tp_as_mapping */
    0,                                    /* tp_hash */
    (ternaryfunc)Dog_call,                /* tp_call */
    0,                                    /* tp_str */
    0,                                    /* tp_getattro */
    0,                                    /* tp_setattro */
    0,                                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                   /* tp_flags */
    "Dog object",                         /* tp_doc */
    0,                                    /* tp_traverse */
    0,                                    /* tp_clear */
    0,                                    /* tp_richcompare */
    0,                                    /* tp_weaklistoffset */
    0,                                    /* tp_iter */
    0,                                    /* tp_iternext */
    0,                                    /* tp_methods */
    Dog_members,                          /* tp_members */
    0,                                    /* tp_getset */
    0,                                    /* tp_base */
    0,                                    /* tp_dict */
    0,                                    /* tp_descr_get */
    0,                                    /* tp_descr_set */
    0,                                    /* tp_dictoffset */
    (initproc)Dog_init,                   /* tp_init */
    0,                                    /* tp_alloc */
    Dog_new,                              /* tp_new */
    0,                                    /* tp_free */
    0,                                    /* tp_is_gc */
};

static PyModuleDef examplemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "example",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_example(void) {
  PyObject *m;
  if (PyType_Ready(&PyDog_Type) < 0)
    return NULL;

  m = PyModule_Create(&examplemodule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&PyDog_Type);
  if (PyModule_AddObject(m, "Dog", (PyObject *)&PyDog_Type) < 0) {
    Py_DECREF(&PyDog_Type);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}