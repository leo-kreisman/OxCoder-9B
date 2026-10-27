"""Host-RAM preflight for an engine load. Without swap, a host that runs short of RAM while a model loads can stall
for minutes (no ssh, no ping) instead of failing. A load that cannot fit stops at once with a message that says what
to do."""
import os

MARGIN = 1 << 30                       # the rest of the load (tokenizer, plans, Python) besides the weights


def available():
    """MemAvailable in bytes, or None when /proc/meminfo cannot be read"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def check(need, what, log=print, avail=None):
    """stop with a clear message when less than need + MARGIN bytes are free. KINDI_HOST_RAM_CHECK=0 skips it."""
    if os.environ.get("KINDI_HOST_RAM_CHECK", "1") != "1":
        return
    if avail is None:
        avail = available()
    if avail is None:
        return
    want = need + MARGIN
    log("host RAM for %s: %.1f GB needed, %.1f GB free" % (what, want / 2 ** 30, avail / 2 ** 30))
    if avail < want:
        raise SystemExit(
            "kindi: %s needs about %.1f GB of free host RAM to load, and %.1f GB is free.\n"
            "  To free RAM: stop the other model servers and large programs on this computer, then start Kindi again.\n"
            "  To load anyway (the computer can become slow or stop answering until the load ends): KINDI_HOST_RAM_CHECK=0"
            % (what, want / 2 ** 30, avail / 2 ** 30))
