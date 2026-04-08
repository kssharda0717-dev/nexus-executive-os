#!/usr/bin/env python3
"""NEXUS Dependency Checker — validates all required libraries are importable.

Run with:  .venv/bin/python check_deps.py
"""

import sys

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"Prefix: {sys.prefix}")
print()

DEPS = [
    # (display_name, import_path, version_attr)
    ("FastAPI", "fastapi", "__version__"),
    ("Starlette", "starlette", "__version__"),
    ("Uvicorn", "uvicorn", "__version__"),
    ("slowapi", "slowapi", None),
    ("slowapi.Limiter", "slowapi", None),
    ("slowapi.errors.RateLimitExceeded", "slowapi.errors", None),
    ("slowapi.util.get_remote_address", "slowapi.util", None),
    ("PyJWT", "jwt", "__version__"),
    ("bcrypt", "bcrypt", "__version__"),
    ("passlib", "passlib", "__version__"),
    ("passlib.hash.bcrypt", "passlib.hash", None),
    ("python-jose", "jose", None),
    ("jose.jwt", "jose.jwt", None),
    ("cryptography", "cryptography", "__version__"),
    ("APScheduler", "apscheduler", None),
    ("Pydantic", "pydantic", "__version__"),
    ("aiosqlite", "aiosqlite", None),
    ("httpx", "httpx", "__version__"),
    ("google.genai", "google.genai", None),
]

passed = 0
failed = 0

for name, mod, ver_attr in DEPS:
    try:
        m = __import__(mod)
        # Walk dotted path for nested modules
        for part in mod.split(".")[1:]:
            m = getattr(m, part)
        ver = getattr(m, ver_attr, "") if ver_attr else ""
        ver_str = f" ({ver})" if ver else ""
        print(f"  ✅ {name:40s}{ver_str}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ {name:40s} → {e}")
        failed += 1

print()
if failed:
    print(f"RESULT: {passed} passed, {failed} FAILED")
    print("Fix: .venv/bin/pip install <missing-package>")
    sys.exit(1)
else:
    print(f"RESULT: All {passed} dependencies OK ✅")
    sys.exit(0)
