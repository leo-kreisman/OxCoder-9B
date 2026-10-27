"""Minimal GGUF (v2/v3) reader: metadata key/values and tensor infos. Tensor data is read by offset."""
import struct

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9


class GGUF(object):
    def __init__(self, path):
        self.path = path
        self.kv = {}
        self.tensors = []                  # [(name, ne (fastest first), type, offset)] in file order
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("%s is not a GGUF file" % path)
            self.version = self._u(f, "I")
            if self.version not in (2, 3):
                raise ValueError("unsupported GGUF version %d" % self.version)
            n_tensors, n_kv = self._u(f, "Q"), self._u(f, "Q")
            for _ in range(n_kv):
                key = self._str(f)
                self.kv[key] = self._value(f, self._u(f, "I"))
            for _ in range(n_tensors):
                name = self._str(f)
                nd = self._u(f, "I")
                ne = [self._u(f, "Q") for _ in range(nd)]
                ty = self._u(f, "I")
                off = self._u(f, "Q")
                self.tensors.append((name, ne, ty, off))
            align = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_start = (pos + align - 1) // align * align

    @staticmethod
    def _u(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    def _str(self, f):
        n = self._u(f, "Q")
        return f.read(n).decode("utf-8", errors="replace")

    def _value(self, f, ty):
        if ty == STRING:
            return self._str(f)
        if ty == ARRAY:
            et, n = self._u(f, "I"), self._u(f, "Q")
            if et == STRING:
                return [self._str(f) for _ in range(n)]
            fmt = _SCALAR[et]
            size = struct.calcsize("<" + fmt)
            return list(struct.unpack("<%d%s" % (n, fmt), f.read(size * n)))
        return self._u(f, _SCALAR[ty])
