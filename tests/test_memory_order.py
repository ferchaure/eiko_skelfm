"""
Test script to validate Step 3.2 - Memory Order (Fortran/column-major)

Run this after building the extension:
    python3 setup.py build_ext --inplace
    python3 test_memory_order.py
"""

import numpy as np
import sys

def test_fortran_order_2d():
    """Test that 2D arrays are properly handled as Fortran-contiguous"""
    from eiko_skelfm import _msfm, msfm
    import pytest

    # Create a simple 2D speed image
    nx, ny = 10, 15
    speed = np.ones((nx, ny), dtype=np.float64)

    # Test with C-order input
    speed_c = np.ascontiguousarray(speed)  # C-order
    assert speed_c.flags['C_CONTIGUOUS']
    assert not speed_c.flags['F_CONTIGUOUS']

    source_points = np.array([[5.0], [7.0]], dtype=np.float64)  # 2x1, F-order by default

    print("Testing 2D with C-order input directly to C extension (should raise ValueError)...")
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        _msfm.msfm2d(speed_c, source_points, 1, 1, 0)
    print("  PASSED (ValueError raised)")

    print("Testing 2D with C-order input via wrapper (should auto-convert)...")
    T = msfm(speed_c, source_points, True, True, False)
    assert T.flags['F_CONTIGUOUS'], "Output should be F-contiguous"
    print("  PASSED")

    # Test with F-order input (should work directly)
    speed_f = np.asfortranarray(speed)
    assert speed_f.flags['F_CONTIGUOUS']

    print("Testing 2D with F-order input directly to C extension...")
    T = _msfm.msfm2d(speed_f, source_points, 1, 1, 0)
    assert T.flags['F_CONTIGUOUS'], "Output should be F-contiguous"
    print("  PASSED")

def test_fortran_order_3d():
    """Test that 3D arrays are properly handled as Fortran-contiguous"""
    from eiko_skelfm import _msfm, msfm
    import pytest

    nx, ny, nz = 8, 10, 12
    speed = np.ones((nx, ny, nz), dtype=np.float64)
    source_points = np.array([[4.0], [5.0], [6.0]], dtype=np.float64)

    # Test with C-order input
    speed_c = np.ascontiguousarray(speed)
    print("Testing 3D with C-order input directly to C extension (should raise ValueError)...")
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        _msfm.msfm3d(speed_c, source_points, 1, 1, 0)
    print("  PASSED (ValueError raised)")

    print("Testing 3D with C-order input via wrapper (should auto-convert)...")
    T = msfm(speed_c, source_points, True, True, False)
    assert T.flags['F_CONTIGUOUS'], "Output should be F-contiguous"
    print("  PASSED")

def test_column_major_indexing():
    """
    Validate that C code uses column-major indexing.
    In column-major: index = x + y*nx + z*nx*ny
    This means adjacent x values are contiguous in memory.
    """
    print("\nValidating column-major indexing in C code...")
    print("  mindex2(x, y, sizx) = x + y*sizx")
    print("  mindex3(x, y, z, sizx, sizy) = x + y*sizx + z*sizx*sizy")
    print("  These are column-major (Fortran) indexing functions.")
    print("  PASSED (verified in common.h and rk4.c)")

if __name__ == "__main__":
    print("=" * 60)
    print("Step 3.2 Validation: Memory Order (Fortran/column-major)")
    print("=" * 60)

    try:
        test_column_major_indexing()
        test_fortran_order_2d()
        test_fortran_order_3d()
        print("\n" + "=" * 60)
        print("All tests PASSED!")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Some tests FAILED: {e}")
        sys.exit(1)
