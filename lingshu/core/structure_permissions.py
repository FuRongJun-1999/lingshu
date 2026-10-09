"""One-time permission checks for the protected service's storage directory."""
import os
from pathlib import Path


def _windows_identity_and_acl(path):
    import ctypes
    from ctypes import wintypes as W
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer = ctypes.c_void_p
    adv.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, ctypes.POINTER(W.HANDLE)]
    adv.GetTokenInformation.argtypes = [W.HANDLE, W.DWORD, pointer, W.DWORD,
                                        ctypes.POINTER(W.DWORD)]
    adv.ConvertSidToStringSidW.argtypes = [pointer, ctypes.POINTER(W.LPWSTR)]
    adv.GetNamedSecurityInfoW.argtypes = [W.LPWSTR, W.DWORD, W.DWORD,
        ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer), pointer,
        ctypes.POINTER(pointer)]
    adv.GetNamedSecurityInfoW.restype = W.DWORD
    adv.GetAce.argtypes = [pointer, W.DWORD, ctypes.POINTER(pointer)]
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer

    def sid_text(sid):
        result = W.LPWSTR()
        if not adv.ConvertSidToStringSidW(sid, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return result.value
        finally:
            kernel.LocalFree(ctypes.cast(result, pointer))

    token = W.HANDLE()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = W.DWORD()
        adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        data = ctypes.create_string_buffer(size.value)
        if not adv.GetTokenInformation(token, 1, data, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        identity = sid_text(ctypes.cast(data, ctypes.POINTER(pointer))[0])
    finally:
        kernel.CloseHandle(token)
    owner, dacl, descriptor = pointer(), pointer(), pointer()
    error = adv.GetNamedSecurityInfoW(str(path), 1, 5, ctypes.byref(owner), None,
                                    ctypes.byref(dacl), None, ctypes.byref(descriptor))
    if error:
        raise ctypes.WinError(error)
    try:
        allowed = {identity, "S-1-5-18", "S-1-5-32-544"}  # service, SYSTEM, Administrators
        if not dacl.value or sid_text(owner) not in allowed:
            raise PermissionError("storage owner must be the service, SYSTEM or Administrators")
        count = ctypes.c_ushort.from_address(dacl.value + 4).value
        for index in range(count):
            ace = pointer()
            if not adv.GetAce(dacl, index, ctypes.byref(ace)):
                raise ctypes.WinError(ctypes.get_last_error())
            kind = ctypes.c_ubyte.from_address(ace.value).value
            if kind == 0:
                # OWNER RIGHTS / CREATOR OWNER refer to the already checked
                # trusted owner; Python's Windows mkdir(0700) may add these.
                owner_aliases = {"S-1-3-0", "S-1-3-4"}
                if sid_text(pointer(ace.value + 8)) not in allowed | owner_aliases:
                    raise PermissionError("storage DACL grants another account access")
            elif kind != 1:  # only ordinary allow/deny ACEs in this dedicated directory
                raise PermissionError("use an explicit service-only storage DACL")
    finally:
        kernel.LocalFree(descriptor)


def check_protected_directory(root, files=()):
    """Fail before opening storage when worker accounts could read it.

    This checks storage access, not the sandbox of a worker. The service and
    workers must run under separate non-administrative OS accounts.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise PermissionError("service storage must be a directory")
    checked = [root]
    for filename in files:
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise PermissionError("protected files must stay inside service storage")
        if path.exists():
            checked.append(path)
    if os.name == "nt":
        for path in checked:
            _windows_identity_and_acl(path)
    elif os.name == "posix":
        stat = root.stat()
        if stat.st_uid != os.geteuid() or stat.st_mode & 0o077:
            raise PermissionError("service must own storage with mode 0700")
    else:
        raise PermissionError("unsupported OS permission model")
    return root
