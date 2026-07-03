#!/usr/bin/env python3
"""
Test script to verify the compiled C library exists after building.

Run after building with CMake:
    cmake -B build -S .
    cmake --build build
    python3 test_lib_build.py
"""

import os
import sys
import subprocess
from pathlib import Path

def test_library_exists():
    """Verify that the compiled library file exists."""
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "build"
    
    # Possible library filenames on different systems
    possible_libs = [
        build_dir / "libeiko_skelfm.a",      # Linux/Unix static lib
        build_dir / "libeiko_skelfm.so",     # Linux shared lib
        build_dir / "libeiko_skelfm.dylib",  # macOS shared lib
        build_dir / "eiko_skelfm.lib",       # Windows static lib
        build_dir / "eiko_skelfm.dll",       # Windows shared lib
    ]
    # Also check inplace python extension
    eiko_dir = project_root / "eiko_skelfm"
    if eiko_dir.exists():
        possible_libs.extend(list(eiko_dir.glob("_msfm.*.so")))
        possible_libs.extend(list(eiko_dir.glob("_msfm.*.pyd")))
        possible_libs.extend(list(eiko_dir.glob("_msfm.*.dylib")))
    
    print("Checking for compiled library...")
    print(f"Build directory: {build_dir}")
    print()
    
    found_libs = [lib for lib in possible_libs if lib.exists()]
    
    if found_libs:
        print("✓ Library file(s) found:")
        for lib in found_libs:
            size_bytes = lib.stat().st_size
            size_kb = size_bytes / 1024
            print(f"  - {lib.name} ({size_kb:.1f} KB)")
    else:
        print("✗ No library files found in build directory")
        print()
        print("Available files in build/:")
        if build_dir.exists():
            for item in sorted(build_dir.iterdir()):
                if item.is_file():
                    print(f"  - {item.name}")
        print()
        print("Please build the library first:")
        print("  cmake -B build -S .")
        print("  cmake --build build")
        assert False, "No library files found in build directory"

if __name__ == "__main__":
    try:
        test_library_exists()
    except AssertionError:
        sys.exit(1)
