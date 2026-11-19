"""Windows/DSH sandbox compatibility shim for the official trainer.

The official `baselines.train` stages output in `tempfile.TemporaryDirectory`
and creates directories with `mode=0o700`.  Under the DSH Windows file sandbox
that mode is enforced as a restrictive ACL, so every later write into the
directory fails with `PermissionError: [WinError 5]`.

Importing this module drops the mode argument from directory creation, which
keeps the staging directory writable.  It must be imported **before** the
official trainer runs (import order only; the trainer itself is untouched).
"""
import os
import pathlib

_real_mkdir = os.mkdir


def _mkdir(path, mode=0o777, *, dir_fd=None):
    if dir_fd is not None:  # not used on Windows, but keep the signature honest
        return _real_mkdir(path, 0o777, dir_fd=dir_fd)
    return _real_mkdir(path, 0o777)


def install() -> None:
    os.mkdir = _mkdir
    for attr in ("_normal_accessor", "_accessor"):
        accessor = getattr(pathlib, attr, None)
        if accessor is not None and hasattr(accessor, "mkdir"):
            setattr(accessor, "mkdir", staticmethod(_mkdir))
    import tempfile

    if hasattr(tempfile, "_os"):
        tempfile._os.mkdir = _mkdir  # noqa: SLF001


install()