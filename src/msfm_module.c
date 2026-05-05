#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "msfm2d.h"
#include "msfm3d.h"

static PyObject* py_msfm2d(PyObject* self, PyObject* args) {
    PyArrayObject *arr_F, *arr_src, *arr_T = NULL, *arr_Y = NULL;
    int use_second, use_cross, return_y = 0;
    npy_intp dims[2];
    double *F, *src, *T, *Y = NULL;
    int n_src, ret;

    if (!PyArg_ParseTuple(args, "O!O!ii|i",
            &PyArray_Type, &arr_F,
            &PyArray_Type, &arr_src,
            &use_second, &use_cross,
            &return_y))
        return NULL;

    if (PyArray_NDIM(arr_F) != 2) {
        PyErr_SetString(PyExc_ValueError, "speed must be 2D");
        return NULL;
    }
    if (PyArray_DTYPE(arr_F)->type_num != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "speed must be float64");
        return NULL;
    }
    if (PyArray_NDIM(arr_src) != 2 || PyArray_DIM(arr_src, 0) != 2) {
        PyErr_SetString(PyExc_ValueError, "source_points must be 2xN");
        return NULL;
    }
    if (PyArray_DTYPE(arr_src)->type_num != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "source_points must be float64");
        return NULL;
    }

    dims[0] = PyArray_DIM(arr_F, 0);
    dims[1] = PyArray_DIM(arr_F, 1);
    F = (double*)PyArray_DATA(arr_F);

    n_src = (int)PyArray_DIM(arr_src, 1);
    src = (double*)PyArray_DATA(arr_src);

    arr_T = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
    if (!arr_T) return NULL;
    T = (double*)PyArray_DATA(arr_T);

    if (return_y) {
        arr_Y = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
        if (!arr_Y) { Py_DECREF(arr_T); return NULL; }
        Y = (double*)PyArray_DATA(arr_Y);
    }

    ret = msfm2d(F, T, Y, src, n_src, (int)dims[0], (int)dims[1],
                 use_second, use_cross);
    if (ret != 0) {
        Py_DECREF(arr_T);
        Py_XDECREF(arr_Y);
        PyErr_SetString(PyExc_RuntimeError, "msfm2d failed");
        return NULL;
    }

    if (return_y) {
        return Py_BuildValue("NN", (PyObject*)arr_T, (PyObject*)arr_Y);
    }
    return (PyObject*)arr_T;
}

static PyObject* py_msfm3d(PyObject* self, PyObject* args) {
    PyArrayObject *arr_F, *arr_src, *arr_T = NULL, *arr_Y = NULL;
    int use_second, use_cross, return_y = 0;
    npy_intp dims[3];
    double *F, *src, *T, *Y = NULL;
    int n_src, ret;

    if (!PyArg_ParseTuple(args, "O!O!ii|i",
            &PyArray_Type, &arr_F,
            &PyArray_Type, &arr_src,
            &use_second, &use_cross,
            &return_y))
        return NULL;

    if (PyArray_NDIM(arr_F) != 3) {
        PyErr_SetString(PyExc_ValueError, "speed must be 3D");
        return NULL;
    }
    if (PyArray_DTYPE(arr_F)->type_num != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "speed must be float64");
        return NULL;
    }
    if (PyArray_NDIM(arr_src) != 2 || PyArray_DIM(arr_src, 0) != 3) {
        PyErr_SetString(PyExc_ValueError, "source_points must be 3xN");
        return NULL;
    }
    if (PyArray_DTYPE(arr_src)->type_num != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "source_points must be float64");
        return NULL;
    }

    dims[0] = PyArray_DIM(arr_F, 0);
    dims[1] = PyArray_DIM(arr_F, 1);
    dims[2] = PyArray_DIM(arr_F, 2);
    F = (double*)PyArray_DATA(arr_F);

    n_src = (int)PyArray_DIM(arr_src, 1);
    src = (double*)PyArray_DATA(arr_src);

    arr_T = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_DOUBLE, 1);
    if (!arr_T) return NULL;
    T = (double*)PyArray_DATA(arr_T);

    if (return_y) {
        arr_Y = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_DOUBLE, 1);
        if (!arr_Y) { Py_DECREF(arr_T); return NULL; }
        Y = (double*)PyArray_DATA(arr_Y);
    }

    ret = msfm3d(F, T, Y, src, n_src, (int)dims[0], (int)dims[1], (int)dims[2],
                 use_second, use_cross);
    if (ret != 0) {
        Py_DECREF(arr_T);
        Py_XDECREF(arr_Y);
        PyErr_SetString(PyExc_RuntimeError, "msfm3d failed");
        return NULL;
    }

    if (return_y) {
        return Py_BuildValue("NN", (PyObject*)arr_T, (PyObject*)arr_Y);
    }
    return (PyObject*)arr_T;
}

static PyMethodDef MsfmMethods[] = {
    {"msfm2d", py_msfm2d, METH_VARARGS, "2D Fast Marching Method"},
    {"msfm3d", py_msfm3d, METH_VARARGS, "3D Fast Marching Method"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef msfmmodule = {
    PyModuleDef_HEAD_INIT,
    "_msfm",
    "Fast Marching Method C extension",
    -1,
    MsfmMethods
};

PyMODINIT_FUNC PyInit__msfm(void) {
    import_array();
    return PyModule_Create(&msfmmodule);
}
