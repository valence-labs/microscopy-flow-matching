"""Utilities for using the PyTorch profiler.
This script can be run as a standalone module to open traces automatically in the browser.
"""

import http.server
import os
import pathlib
import socket
import socketserver
from functools import partial

import torch


class _PerfettoServer(http.server.SimpleHTTPRequestHandler):
    """Handles requests from `ui.perfetto.dev` for the `trace.json`.
    Forked from github.com/jax-ml/jax (Apache 2.0 license)."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        return super().end_headers()

    def do_GET(self):
        self.server.last_request = self.path  # type: ignore
        return super().do_GET()

    def do_POST(self):
        self.send_error(404, "File not found")


def host_perfetto_trace_file(path: os.PathLike | str):
    # ui.perfetto.dev looks for files hosted on `127.0.0.1:9001`. We set up a
    # TCP server that is hosting the `perfetto_trace.json.gz` file.
    # Forked from github.com/jax-ml/jax (Apache 2.0 license)
    port = 9001
    orig_directory = pathlib.Path.cwd()
    directory, filename = os.path.split(path)
    try:
        os.chdir(directory)
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", port), _PerfettoServer) as httpd:
            url = f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{filename}"
            print(f"Open URL in browser: {url}")

            # Once ui.perfetto.dev acquires trace.json from this server we can close
            # it down.
            while httpd.__dict__.get("last_request") != "/" + filename:
                httpd.handle_request()
    finally:
        os.chdir(orig_directory)


def on_trace_ready(
    prof: torch.profiler.profile,
    path: str = "trace.json",
    verbose: bool = False,
    open_browser: bool = False,
):
    if verbose:
        avgs = prof.key_averages()
        print(f"Profile:\n{avgs}")
    prof.export_chrome_trace(path)
    print("-------------------------------------------------------------")
    print(f"Trace saved to {os.path.abspath(path)}")
    print("You can open the trace in chrome://tracing or ui.perfetto.dev")
    print("-------------------------------------------------------------")

    if open_browser:
        print("Opening trace in browser...")
        print("(Program will block until you open the link)")
        host_perfetto_trace_file(os.path.abspath(path))
    else:
        print("Run `python -m utils.profiler` to open the trace automatically.")


class DummyProfiler:
    """Dummy profiler object to use on non-zero ranks in DDP training."""

    def __init__(self, *args, **kwargs): ...
    def __enter__(self): ...
    def __exit__(self, _, __, ___): ...
    def step(self): ...
    def start(self): ...
    def stop(self): ...
    def set_custom_trace_id_callback(self, callback): ...
    def get_trace_id(self): ...


def get_profiler(
    *,
    wait: int = 5,
    warmup: int = 5,
    active: int = 5,
    repeat: int = 1,
    save_dir: str | None = None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    open_browser: bool = False,
    return_dummy: bool = False,
    verbose: bool = False,
) -> torch.profiler.profile | DummyProfiler:
    """Gets a properly configured pytorch profiler. You can optionally return
    a dummy profiler to use on non-zero ranks in DDP training.

    IMPORTANT: if you set `open_browser=True`, the profiler will block the main
    process until you open the trace in the browser.

    Example usage:
    ```python
    # only profile the zero rank process in DDP training
    prof = get_profiler(return_dummy=not fabric.is_global_zero)
    prof.start()
    for batch in train_loader:
        ...
        prof.step()
    prof.stop()
    ```

    Args:
        wait: Number of steps to wait before recording.
        warmup: Number of steps to warmup before recording.
        active: Number of steps to record.
        repeat: Number of times to repeat the recording.
        save_dir: Directory to save the trace file. If None, defaults to cwd.
        open_browser: Whether to open the trace in the browser.
        return_dummy: Whether to return a dummy profiler.

    Returns:
        A pytorch profiler if return_dummy is False, otherwise a dummy profiler.
    """
    if return_dummy:
        return DummyProfiler()

    if save_dir is None:
        save_dir = os.getcwd()

    path = f"trace_{socket.gethostname()}.json"
    path = os.path.join(save_dir, path)

    s = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    f = partial(on_trace_ready, path=path, open_browser=open_browser, verbose=verbose)
    prof = torch.profiler.profile(
        schedule=s,
        on_trace_ready=f,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )
    return prof


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    if args.path:
        path = args.path
    else:
        path = input("Enter the path to the trace file: ")

    host_perfetto_trace_file(path)
