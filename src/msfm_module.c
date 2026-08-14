#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "msfm2d.h"
#include "msfm3d.h"
#include "rk4.h"

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
    if (!PyArray_IS_F_CONTIGUOUS(arr_F)) {
        PyErr_SetString(PyExc_ValueError, "speed must be Fortran-contiguous (use np.asfortranarray)");
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
    if (!PyArray_IS_F_CONTIGUOUS(arr_src)) {
        PyErr_SetString(PyExc_ValueError, "source_points must be Fortran-contiguous (use np.asfortranarray)");
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
    if (!PyArray_IS_F_CONTIGUOUS(arr_F)) {
        PyErr_SetString(PyExc_ValueError, "speed must be Fortran-contiguous (use np.asfortranarray)");
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
    if (!PyArray_IS_F_CONTIGUOUS(arr_src)) {
        PyErr_SetString(PyExc_ValueError, "source_points must be Fortran-contiguous (use np.asfortranarray)");
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

static PyObject* py_rk4_step_2d(PyObject* self, PyObject* args) {
    PyArrayObject *arr_grad, *arr_start;
    double step_size;

    if (!PyArg_ParseTuple(args, "O!O!d",
            &PyArray_Type, &arr_grad,
            &PyArray_Type, &arr_start,
            &step_size))
        return NULL;

    if (PyArray_NDIM(arr_grad) != 3 || PyArray_DIM(arr_grad, 2) != 2) {
        PyErr_SetString(PyExc_ValueError, "gradient must be (nx, ny, 2)");
        return NULL;
    }
    if (!PyArray_IS_F_CONTIGUOUS(arr_grad)) {
        PyErr_SetString(PyExc_ValueError, "gradient must be Fortran-contiguous");
        return NULL;
    }

    double *grad = (double*)PyArray_DATA(arr_grad);
    double *start = (double*)PyArray_DATA(arr_start);
    int dims[2] = { (int)PyArray_DIM(arr_grad, 0), (int)PyArray_DIM(arr_grad, 1) };

    double next[2];
    int ok = RK4STEP_2D(grad, dims, start, next, step_size);
    if (!ok) {
        Py_RETURN_NONE;
    }

    npy_intp out_dims[1] = {2};
    PyObject *out = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    double *out_data = (double*)PyArray_DATA((PyArrayObject*)out);
    out_data[0] = next[0];
    out_data[1] = next[1];
    return out;
}

static PyObject* py_rk4_step_3d(PyObject* self, PyObject* args) {
    PyArrayObject *arr_grad, *arr_start;
    double step_size;

    if (!PyArg_ParseTuple(args, "O!O!d",
            &PyArray_Type, &arr_grad,
            &PyArray_Type, &arr_start,
            &step_size))
        return NULL;

    if (PyArray_NDIM(arr_grad) != 4 || PyArray_DIM(arr_grad, 3) != 3) {
        PyErr_SetString(PyExc_ValueError, "gradient must be (nx, ny, nz, 3)");
        return NULL;
    }
    if (!PyArray_IS_F_CONTIGUOUS(arr_grad)) {
        PyErr_SetString(PyExc_ValueError, "gradient must be Fortran-contiguous");
        return NULL;
    }

    double *grad = (double*)PyArray_DATA(arr_grad);
    double *start = (double*)PyArray_DATA(arr_start);
    int dims[3] = { (int)PyArray_DIM(arr_grad, 0), (int)PyArray_DIM(arr_grad, 1), (int)PyArray_DIM(arr_grad, 2) };

    double next[3];
    int ok = RK4STEP_3D(grad, dims, start, next, step_size);
    if (!ok) {
        Py_RETURN_NONE;
    }

    npy_intp out_dims[1] = {3};
    PyObject *out = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    double *out_data = (double*)PyArray_DATA((PyArrayObject*)out);
    out_data[0] = next[0];
    out_data[1] = next[1];
    out_data[2] = next[2];
    return out;
}

static PyMethodDef MsfmMethods[] = {
    {"msfm2d", py_msfm2d, METH_VARARGS, "2D Fast Marching Method"},
    {"msfm3d", py_msfm3d, METH_VARARGS, "3D Fast Marching Method"},
    {"rk4_step_2d", py_rk4_step_2d, METH_VARARGS, "2D RK4 step"},
    {"rk4_step_3d", py_rk4_step_3d, METH_VARARGS, "3D RK4 step"},
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
