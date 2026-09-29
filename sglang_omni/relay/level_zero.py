# SPDX-License-Identifier: Apache-2.0
"""Level-Zero IPC primitives for Intel GPU device-to-device stage transport.

Level Zero exports a device allocation as a dma_buf descriptor, and a descriptor
only reaches another process over a Unix socket carrying SCM_RIGHTS. An exporting
relay therefore serves its handle from a rendezvous socket whose abstract name
travels in the control-plane metadata, and an importing relay maps the peer
allocation into its own Level-Zero context.

Torch owns the device memory on both sides. This module never allocates: it reads
the segment that a torch pointer belongs to, and hands an imported peer mapping
back to torch through DLPack, so both the staging copy and the peer copy stay
ordered on the torch stream that issues them.
"""
from __future__ import annotations

import ctypes
import logging
import os
import socket
import struct
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

LEVEL_ZERO_LOADER = "libze_loader.so.1"
ZE_RESULT_SUCCESS = 0
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_MAX_IPC_HANDLE_SIZE = 64

# The Intel driver keeps the dma_buf descriptor in the first four bytes of an IPC
# handle and its own metadata in the remaining bytes, so an importer replaces only
# those bytes with the descriptor its own process received.
IPC_HANDLE_DESCRIPTOR_BYTES = 4

DLPACK_DEVICE_ONEAPI = 14
DLPACK_TYPE_CODE_UINT = 1
DLPACK_UINT8_BITS = 8

RENDEZVOUS_BACKLOG = 16
RENDEZVOUS_POLL_S = 0.2
RENDEZVOUS_SHUTDOWN_S = 5.0
HANDLE_REQUEST_TIMEOUT_S = 5.0
HANDLE_REQUEST_LIMIT_BYTES = 512
DISABLE_ENV = "SGLANG_OMNI_XPU_LEVEL_ZERO_IPC"


class LevelZeroError(RuntimeError):
    """A Level-Zero entry point reported a failure status."""


class ZeContextDescriptor(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class ZeIpcMemHandle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * ZE_MAX_IPC_HANDLE_SIZE)]


@lru_cache(maxsize=1)
def load_level_zero() -> ctypes.CDLL:
    """Load the Level-Zero loader, declare the entry points, and initialize it.

    Torch comes up first because this module only ever exports memory that torch
    allocated, and because a loader that has not yet served the XPU runtime
    rejects zeInit with an unsupported-version status.
    """
    torch.xpu.init()
    library = ctypes.CDLL(LEVEL_ZERO_LOADER)
    void_pointer = ctypes.c_void_p
    count_pointer = ctypes.POINTER(ctypes.c_uint32)
    handle_pointer = ctypes.POINTER(ctypes.c_void_p)

    argument_types: dict[str, list[type]] = {
        "zeInit": [ctypes.c_uint32],
        "zeDriverGet": [count_pointer, handle_pointer],
        "zeDeviceGet": [void_pointer, count_pointer, handle_pointer],
        "zeContextCreate": [
            void_pointer,
            ctypes.POINTER(ZeContextDescriptor),
            handle_pointer,
        ],
        "zeContextDestroy": [void_pointer],
        "zeMemGetAddressRange": [
            void_pointer,
            void_pointer,
            handle_pointer,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        "zeMemGetIpcHandle": [
            void_pointer,
            void_pointer,
            ctypes.POINTER(ZeIpcMemHandle),
        ],
        "zeMemPutIpcHandle": [void_pointer, ZeIpcMemHandle],
        "zeMemOpenIpcHandle": [
            void_pointer,
            void_pointer,
            ZeIpcMemHandle,
            ctypes.c_uint32,
            handle_pointer,
        ],
        "zeMemCloseIpcHandle": [void_pointer, void_pointer],
    }
    for name, types in argument_types.items():
        entry_point = getattr(library, name)
        entry_point.argtypes = types
        entry_point.restype = ctypes.c_int

    status = library.zeInit(0)
    if status != ZE_RESULT_SUCCESS:
        raise LevelZeroError(f"zeInit failed with status 0x{status & 0xFFFFFFFF:x}")
    else:
        pass
    return library


def check_status(entry_point_name: str, status: int) -> None:
    if status != ZE_RESULT_SUCCESS:
        raise LevelZeroError(
            f"{entry_point_name} failed with status 0x{status & 0xFFFFFFFF:x}"
        )
    else:
        pass


@lru_cache(maxsize=1)
def level_zero_device_count() -> int:
    """Count the devices the Level-Zero driver exposes to this process."""
    library = load_level_zero()
    driver_count = ctypes.c_uint32(0)
    check_status("zeDriverGet", library.zeDriverGet(ctypes.byref(driver_count), None))
    if driver_count.value == 0:
        return 0
    else:
        pass
    drivers = (ctypes.c_void_p * driver_count.value)()
    check_status(
        "zeDriverGet", library.zeDriverGet(ctypes.byref(driver_count), drivers)
    )
    device_count = ctypes.c_uint32(0)
    check_status(
        "zeDeviceGet",
        library.zeDeviceGet(drivers[0], ctypes.byref(device_count), None),
    )
    return int(device_count.value)


@lru_cache(maxsize=1)
def is_level_zero_ipc_available() -> bool:
    """Whether this process can move stage payloads over Level-Zero IPC.

    A Level-Zero device order that disagrees with torch disables the transport:
    an IPC handle is opened against a device index chosen by torch, so a mismatched
    order would map a peer allocation onto the wrong card.
    """
    if os.getenv(DISABLE_ENV, "1") == "0":
        logger.info(f"Level-Zero IPC disabled by {DISABLE_ENV}")
        return False
    elif not torch.xpu.is_available():
        return False
    else:
        pass

    # A loader or runtime that cannot come up leaves the caller on host staging, so
    # every startup failure here is a fallback rather than a stage failure.
    try:
        torch_device_count = torch.xpu.device_count()
        driver_device_count = level_zero_device_count()
    except (OSError, RuntimeError) as error:
        logger.info(f"Level-Zero IPC unavailable: {error}")
        return False

    if driver_device_count != torch_device_count:
        logger.warning(
            f"Level-Zero IPC disabled: the driver exposes {driver_device_count} "
            f"devices but torch sees {torch_device_count}, so device indices "
            f"cannot be trusted to match"
        )
        return False
    else:
        pass
    return True


@dataclass(frozen=True, kw_only=True)
class ExportedSegment:
    """One exported Level-Zero allocation and the descriptor that shares it."""

    base_pointer: int
    segment_bytes: int
    handle_blob: bytes
    file_descriptor: int


class LevelZeroContext:
    """A private Level-Zero context used only to export and import IPC handles.

    The Intel driver resolves a device pointer through a driver-wide allocation
    map, so this context can export memory that torch allocated in its own.
    """

    def __init__(self) -> None:
        self.library = load_level_zero()
        driver_count = ctypes.c_uint32(0)
        check_status(
            "zeDriverGet", self.library.zeDriverGet(ctypes.byref(driver_count), None)
        )
        if driver_count.value == 0:
            raise LevelZeroError("no Level-Zero driver is available")
        else:
            pass
        drivers = (ctypes.c_void_p * driver_count.value)()
        check_status(
            "zeDriverGet", self.library.zeDriverGet(ctypes.byref(driver_count), drivers)
        )
        self.driver = drivers[0]

        device_count = ctypes.c_uint32(0)
        check_status(
            "zeDeviceGet",
            self.library.zeDeviceGet(self.driver, ctypes.byref(device_count), None),
        )
        devices = (ctypes.c_void_p * device_count.value)()
        check_status(
            "zeDeviceGet",
            self.library.zeDeviceGet(self.driver, ctypes.byref(device_count), devices),
        )
        self.devices: tuple[int, ...] = tuple(
            int(devices[index]) for index in range(device_count.value)
        )

        descriptor = ZeContextDescriptor(ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
        context = ctypes.c_void_p()
        check_status(
            "zeContextCreate",
            self.library.zeContextCreate(
                self.driver, ctypes.byref(descriptor), ctypes.byref(context)
            ),
        )
        self.context = context

    def export_segment(self, device_pointer: int) -> ExportedSegment:
        """Export the whole allocation that a device pointer falls inside."""
        base = ctypes.c_void_p()
        segment_bytes = ctypes.c_size_t(0)
        check_status(
            "zeMemGetAddressRange",
            self.library.zeMemGetAddressRange(
                self.context,
                ctypes.c_void_p(device_pointer),
                ctypes.byref(base),
                ctypes.byref(segment_bytes),
            ),
        )
        handle = ZeIpcMemHandle()
        check_status(
            "zeMemGetIpcHandle",
            self.library.zeMemGetIpcHandle(self.context, base, ctypes.byref(handle)),
        )
        handle_blob = bytes(bytearray(handle.data))
        return ExportedSegment(
            base_pointer=int(base.value),
            segment_bytes=int(segment_bytes.value),
            handle_blob=handle_blob,
            file_descriptor=struct.unpack_from("<i", handle_blob)[0],
        )

    def release_segment(self, segment: ExportedSegment) -> None:
        handle = ZeIpcMemHandle()
        handle.data = (ctypes.c_ubyte * ZE_MAX_IPC_HANDLE_SIZE)(*segment.handle_blob)
        check_status(
            "zeMemPutIpcHandle",
            self.library.zeMemPutIpcHandle(self.context, handle),
        )

    def open_peer_segment(
        self,
        handle_blob: bytes,
        file_descriptor: int,
        device_index: int,
    ) -> int:
        """Map a peer allocation, returning its pointer in this process."""
        if device_index >= len(self.devices):
            raise LevelZeroError(
                f"Level-Zero device {device_index} is outside the {len(self.devices)} "
                f"devices this process can see"
            )
        else:
            pass
        local_blob = bytearray(handle_blob)
        struct.pack_into("<i", local_blob, 0, file_descriptor)
        handle = ZeIpcMemHandle()
        handle.data = (ctypes.c_ubyte * ZE_MAX_IPC_HANDLE_SIZE)(*local_blob)
        peer_pointer = ctypes.c_void_p()
        check_status(
            "zeMemOpenIpcHandle",
            self.library.zeMemOpenIpcHandle(
                self.context,
                ctypes.c_void_p(self.devices[device_index]),
                handle,
                0,
                ctypes.byref(peer_pointer),
            ),
        )
        return int(peer_pointer.value)

    def close_peer_segment(self, peer_pointer: int) -> None:
        check_status(
            "zeMemCloseIpcHandle",
            self.library.zeMemCloseIpcHandle(
                self.context, ctypes.c_void_p(peer_pointer)
            ),
        )

    def destroy(self) -> None:
        if self.context is None:
            return
        else:
            pass
        check_status("zeContextDestroy", self.library.zeContextDestroy(self.context))
        self.context = None


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


# A wrapped tensor borrows memory that the Level-Zero mapping owns, so its DLPack
# descriptor carries no deleter and has to outlive every tensor torch builds from
# it. One entry per imported peer pool keeps the list bounded.
DLPACK_DESCRIPTORS: list[tuple[DLManagedTensor, ctypes.Array, ctypes.Array]] = []


def wrap_device_bytes(
    device_pointer: int,
    num_bytes: int,
    device_index: int,
) -> torch.Tensor:
    """Expose a Level-Zero device pointer to torch as a uint8 tensor.

    The tensor names the importing device, which is the device that runs the
    copies reading it, not the device that physically holds the memory.
    """
    shape = (ctypes.c_int64 * 1)(num_bytes)
    strides = (ctypes.c_int64 * 1)(1)
    descriptor = DLManagedTensor()
    descriptor.dl_tensor.data = ctypes.c_void_p(device_pointer)
    descriptor.dl_tensor.device = DLDevice(DLPACK_DEVICE_ONEAPI, device_index)
    descriptor.dl_tensor.ndim = 1
    descriptor.dl_tensor.dtype = DLDataType(DLPACK_TYPE_CODE_UINT, DLPACK_UINT8_BITS, 1)
    descriptor.dl_tensor.shape = shape
    descriptor.dl_tensor.strides = strides
    descriptor.dl_tensor.byte_offset = 0
    descriptor.manager_ctx = None
    descriptor.deleter = None
    DLPACK_DESCRIPTORS.append((descriptor, shape, strides))

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(ctypes.byref(descriptor), b"dltensor", None)
    return torch.from_dlpack(capsule)


class HandleRendezvous:
    """Serves one exported segment to importing peers over a Unix socket.

    A dma_buf descriptor cannot travel through the control plane, so importers
    collect it here with SCM_RIGHTS. The socket lives in the abstract namespace so
    that it leaves no file behind when the process exits.
    """

    def __init__(self, pool_id: str, segment: ExportedSegment) -> None:
        self.pool_id = pool_id
        self.segment = segment
        self.socket_name = (
            f"sglang-omni-level-zero-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        )
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind("\0" + self.socket_name)
        self.listener.listen(RENDEZVOUS_BACKLOG)
        self.listener.settimeout(RENDEZVOUS_POLL_S)
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self.serve,
            name=f"level-zero-ipc-{self.socket_name}",
            daemon=True,
        )
        self.thread.start()

    def serve(self) -> None:
        while not self.stopped.is_set():
            try:
                connection, _peer = self.listener.accept()
            except TimeoutError:
                continue
            except OSError as error:
                if self.stopped.is_set():
                    return
                else:
                    raise LevelZeroError(
                        f"Level-Zero IPC rendezvous {self.socket_name} failed: {error}"
                    ) from error
            with connection:
                connection.settimeout(HANDLE_REQUEST_TIMEOUT_S)
                try:
                    self.send_handle(connection)
                except (OSError, LevelZeroError) as error:
                    # One importer giving up must not stop the pool from reaching
                    # the rest, and this thread is the only one serving them.
                    logger.warning(
                        f"Level-Zero IPC rendezvous {self.socket_name} could not "
                        f"serve a peer: {error}"
                    )

    def send_handle(self, connection: socket.socket) -> None:
        requested_pool_id = receive_message(connection).decode("utf-8")
        if requested_pool_id != self.pool_id:
            logger.warning(
                f"Level-Zero IPC rendezvous {self.socket_name} rejected a request "
                f"for pool {requested_pool_id!r}"
            )
            return
        else:
            pass
        socket.send_fds(
            connection, [self.segment.handle_blob], [self.segment.file_descriptor]
        )

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=RENDEZVOUS_SHUTDOWN_S)
        self.listener.close()


def receive_exactly(connection: socket.socket, num_bytes: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < num_bytes:
        chunk = connection.recv(num_bytes - len(chunks))
        if not chunk:
            raise LevelZeroError(
                f"Level-Zero IPC peer closed the connection after "
                f"{len(chunks)} of {num_bytes} bytes"
            )
        else:
            pass
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(struct.pack("<I", len(payload)) + payload)


def receive_message(connection: socket.socket) -> bytes:
    (length,) = struct.unpack("<I", receive_exactly(connection, 4))
    if length > HANDLE_REQUEST_LIMIT_BYTES:
        raise LevelZeroError(
            f"Level-Zero IPC request claims {length} bytes, over the "
            f"{HANDLE_REQUEST_LIMIT_BYTES} byte limit"
        )
    else:
        pass
    return receive_exactly(connection, length)


def fetch_peer_segment(socket_name: str, pool_id: str) -> tuple[bytes, int]:
    """Collect an exported handle and its descriptor from the owning process."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect("\0" + socket_name)
        send_message(connection, pool_id.encode("utf-8"))
        handle_blob, descriptors, _flags, _address = socket.recv_fds(
            connection, ZE_MAX_IPC_HANDLE_SIZE, 1
        )
    if len(descriptors) != 1:
        raise LevelZeroError(
            f"Level-Zero IPC rendezvous {socket_name} returned "
            f"{len(descriptors)} descriptors for pool {pool_id!r}"
        )
    elif len(handle_blob) != ZE_MAX_IPC_HANDLE_SIZE:
        raise LevelZeroError(
            f"Level-Zero IPC rendezvous {socket_name} returned a "
            f"{len(handle_blob)} byte handle for pool {pool_id!r}"
        )
    else:
        pass
    return handle_blob, descriptors[0]
