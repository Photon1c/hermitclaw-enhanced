"""Sandboxed shell — the agent can run commands, but only inside environment/."""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from urllib.error import URLError
from urllib.parse import urlparse

from hermitclaw.config import config

logger = logging.getLogger("hermitclaw.tools")

OLLAMA_WEB_SEARCH_URL = "https://ollama.com/api/web_search"
OLLAMA_WEB_FETCH_URL = "https://ollama.com/api/web_fetch"

# Commands that should never be run (checked as prefixes after stripping)
BLOCKED_PREFIXES = [
    "sudo",
    "su ",
    "rm -rf /",
    "chmod",
    "chown",
    "kill",
    "pkill",
    "curl",
    "wget",
    "nc ",
    "ncat",
    "ssh",
    "scp",
    "sftp",
    "node",
    "ruby",
    "perl",
    "bash",
    "sh ",
    "zsh",
    "export",
    "source",
    "eval",
    "exec",
    "mount",
    "umount",
    "dd ",
    "mkfs",
    "fdisk",
    "apt",
    "brew",
    "npm",
    "yarn",
    "open ",
    "xdg-open",
]

# Path to the Python sandbox wrapper
_SANDBOX = os.path.join(os.path.dirname(__file__), "pysandbox.py")


def _venv_dir(env_root: str) -> str:
    """Path to the crab's virtual environment."""
    return os.path.join(os.path.realpath(env_root), ".venv")


def _venv_python(env_root: str) -> str:
    """Path to the venv's Python interpreter."""
    return os.path.join(_venv_dir(env_root), "bin", "python")


def _venv_bin(env_root: str) -> str:
    """Path to the venv's bin directory."""
    return os.path.join(_venv_dir(env_root), "bin")


def ensure_venv(env_root: str):
    """Create the crab's venv if it doesn't exist. Called once on startup."""
    venv = _venv_dir(env_root)
    if os.path.isfile(_venv_python(env_root)):
        return
    logger.info(f"Creating crab venv at {venv}...")
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "venv", venv, "--python", sys.executable, "--seed", "pip"],
            capture_output=True,
            timeout=30,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "venv", venv], capture_output=True, timeout=30
        )
    # Ensure 'python' exists (some venvs only have python3)
    py_bin = os.path.join(venv, "bin")
    python3_path = os.path.join(py_bin, "python3")
    python_path = os.path.join(py_bin, "python")
    if os.path.isfile(python3_path) and not os.path.isfile(python_path):
        try:
            os.symlink("python3", python_path)
        except OSError:
            pass
    logger.info("Crab venv created.")


def _is_safe_command(command: str) -> str | None:
    """Return an error message if the command is unsafe, else None."""
    stripped = command.strip()

    if not stripped:
        return "Blocked: empty command."

    # Block dangerous command prefixes
    for prefix in BLOCKED_PREFIXES:
        if stripped.startswith(prefix):
            return f"Blocked: '{prefix}' commands are not allowed."

    # Block parent directory traversal — check actual path tokens, not content strings.
    # Split on whitespace and strip shell operators to find path-like tokens.
    for token in stripped.split():
        clean = token.lstrip("><=|;&(")
        if clean == ".." or clean.startswith("../") or "/.." in clean:
            return "Blocked: '..' path traversal is not allowed in commands."

    # Block shell escape tricks
    if "`" in stripped:
        return "Blocked: backtick command substitution is not allowed."
    if "$(" in stripped:
        return "Blocked: command substitution $() is not allowed."
    if "${" in stripped:
        return "Blocked: variable expansion ${} is not allowed."
    if "~" in stripped:
        return "Blocked: '~' (home expansion) is not allowed."

    # Block absolute paths — only relative paths from environment/ are allowed.
    # Check each whitespace-separated token; strip leading shell operators.
    # Only flag tokens where / is followed by a word char (actual paths like /usr/bin),
    # not markup like /> or /' or /" which appear in XML/HTML/SVG content.
    import re

    for token in stripped.split():
        clean = token.lstrip("><=|;&(")
        if re.match(r"/[A-Za-z0-9_]", clean) and not clean.startswith("/dev/null"):
            return "Blocked: absolute paths are not allowed. Use relative paths only."

    return None


def _rewrite_python_cmd(command: str, env_root: str) -> str | None:
    """If command is a python invocation, rewrite to run through the sandbox.

    Uses the crab's venv python so installed packages are available.
    Returns the rewritten command, or None if it's not a python command.
    """
    stripped = command.strip()
    if stripped.startswith("python3"):
        rest = stripped[7:]
    elif stripped.startswith("python"):
        rest = stripped[6:]
    else:
        return None
    real_root = os.path.realpath(env_root)
    python = (
        _venv_python(env_root)
        if os.path.isfile(_venv_python(env_root))
        else sys.executable
    )
    return (
        f"{shlex.quote(python)} {shlex.quote(_SANDBOX)} {shlex.quote(real_root)}{rest}"
    )


def _rewrite_script_cmd(command: str, env_root: str) -> str | None:
    """Route ./script.py through sandbox so network etc. is blocked."""
    stripped = command.strip()
    if stripped.startswith("./") and stripped.endswith(".py"):
        script = stripped[2:].split()[0]  # ./foo.py or ./foo.py arg1
        rest = stripped[2 + len(script) :].strip()  # any args after script
        python = (
            _venv_python(env_root)
            if os.path.isfile(_venv_python(env_root))
            else sys.executable
        )
        real_root = os.path.realpath(env_root)
        return f"{shlex.quote(python)} {shlex.quote(_SANDBOX)} {shlex.quote(real_root)} {shlex.quote(script)}{' ' + rest if rest else ''}"
    return None


def _rewrite_pip_cmd(command: str, env_root: str) -> str | None:
    """If command is pip/uv pip, rewrite to use the venv. Returns rewritten cmd or None."""
    stripped = command.strip()
    if stripped.startswith("uv pip "):
        # Route through venv python
        rest = stripped[7:]  # after "uv pip "
        uv = shutil.which("uv") or "uv"
        return f"{shlex.quote(uv)} pip {rest} --python {shlex.quote(_venv_python(env_root))}"
    if stripped.startswith("pip install") or stripped.startswith("pip3 install"):
        # Use venv pip
        return f"{shlex.quote(_venv_python(env_root))} -m pip {stripped[stripped.index('install') :]}"
    return None


def run_command(command: str, env_root: str) -> str:
    """Run a shell command sandboxed to the environment/ folder."""
    real_root = os.path.realpath(env_root)

    # Safety check (runs on original command before any rewriting)
    err = _is_safe_command(command)
    if err:
        return err

    # Route python commands through the sandbox wrapper
    rewritten = _rewrite_python_cmd(command, env_root)
    if rewritten is not None:
        command = rewritten

    # Route ./script.py through sandbox (otherwise shebang bypasses pysandbox)
    script_rewritten = _rewrite_script_cmd(command, env_root)
    if script_rewritten is not None:
        command = script_rewritten
    # Route pip/uv pip through the venv
    pip_rewritten = _rewrite_pip_cmd(command, env_root)
    if pip_rewritten is not None:
        command = pip_rewritten

    # Include venv bin in PATH so installed tools are available
    vbin = _venv_bin(env_root)
    venv_path = f"{vbin}:/usr/bin:/bin" if os.path.isdir(vbin) else "/usr/bin:/bin"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=real_root,
            capture_output=True,
            text=True,
            timeout=60,  # longer timeout for pip installs
            env={
                "HOME": real_root,
                "PATH": venv_path,
                "TMPDIR": real_root,
                "LANG": "en_US.UTF-8",
                "VIRTUAL_ENV": _venv_dir(env_root),
            },
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr

        if not output.strip():
            output = "(no output)"

        # Truncate very long output
        if len(output) > 3000:
            output = output[:3000] + "\n...(truncated)"

        return output

    except subprocess.TimeoutExpired:
        return "Error: command timed out (15s limit)"
    except Exception as e:
        return f"Error: {e}"


def ollama_web_search(query: str, max_results: int = 5) -> str:
    """Call Ollama cloud web search API. Requires OLLAMA_API_KEY."""
    api_key = config.get("ollama_api_key")
    if not api_key:
        return "Error: OLLAMA_API_KEY is required for web search. Get one at https://ollama.com/settings/keys"
    try:
        data = json.dumps(
            {"query": query, "max_results": min(max_results, 10)}
        ).encode()
        req = urllib.request.Request(
            OLLAMA_WEB_SEARCH_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode())
        lines = []
        for r in out.get("results", []):
            lines.append(f"**{r.get('title', '')}**")
            lines.append(f"URL: {r.get('url', '')}")
            lines.append(r.get("content", "")[:2000])
            lines.append("")
        return "\n".join(lines).strip()[:8000] or "No results found."
    except URLError as e:
        return f"Error: {e.reason}"
    except Exception as e:
        return f"Error: {e}"


def ollama_web_fetch(url: str) -> str:
    """Call Ollama cloud web fetch API. Requires OLLAMA_API_KEY."""
    api_key = config.get("ollama_api_key")
    if not api_key:
        return "Error: OLLAMA_API_KEY is required for web fetch. Get one at https://ollama.com/settings/keys"
    try:
        data = json.dumps({"url": url}).encode()
        req = urllib.request.Request(
            OLLAMA_WEB_FETCH_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode())
        title = out.get("title", "")
        content = out.get("content", "")[:6000]
        return f"**{title}**\n\n{content}" if title else content
    except URLError as e:
        return f"Error: {e.reason}"
    except Exception as e:
        return f"Error: {e}"


def fetch_url(url: str, max_chars: int = 12000, timeout: int = 15) -> str:
    """Fetch a URL and return its content (for research). Runs in main process, not sandbox."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Error: Only http and https URLs are allowed."
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "HermitClaw/1.0 (research)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...(truncated)"
        # Simple HTML-to-text: strip tags, collapse whitespace
        text = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] if len(text) > max_chars else text or body[:max_chars]
    except URLError as e:
        return f"Error fetching URL: {e.reason}"
    except Exception as e:
        return f"Error: {e}"


def _resolveObservePath(path: str) -> str:
    """Resolve a path that may use ~ or be relative to home."""
    path = path.strip()
    # Resolve ~ to home directory
    if path.startswith("~"):
        home = os.path.expanduser("~")
        if path == "~":
            return home
        return os.path.join(home, path[2:].lstrip("/"))
    # If it's already an absolute path, validate it's under home
    if os.path.isabs(path):
        home = os.path.expanduser("~")
        if not path.startswith(home):
            return home  # Fallback to home if outside
        return path
    # Relative path - reject for safety (must use ~ or absolute)
    return ""


def _getDirectoryTree(path: str, max_depth: int = 3, max_items: int = 50) -> str:
    """Get a directory tree structure (read-only)."""
    import os

    home = os.path.expanduser("~")
    resolved = _resolveObservePath(path)
    if not resolved or not os.path.isdir(resolved):
        return f"Error: Cannot access {path} - invalid or inaccessible path"

    # Security: ensure resolved path is under home
    if not resolved.startswith(home):
        return f"Error: Access denied - {path} is outside allowed area"

    lines = []
    try:
        entries = sorted(
            os.listdir(resolved),
            key=lambda x: (not os.path.isdir(os.path.join(resolved, x)), x),
        )
        for i, entry in enumerate(entries[:max_items]):
            if i >= max_items:
                lines.append(f"... and {len(entries) - max_items} more items")
                break
            entry_path = os.path.join(resolved, entry)
            if os.path.isdir(entry_path):
                # Count items in directory
                try:
                    subcount = len(os.listdir(entry_path))
                    lines.append(f"📁 {entry}/ ({subcount} items)")
                except:
                    lines.append(f"📁 {entry}/")
            else:
                size = os.path.getsize(entry_path)
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"📄 {entry} ({size_str})")
    except PermissionError:
        return f"Error: Permission denied for {path}"
    except Exception as e:
        return f"Error: {e}"

    return "\n".join(lines) if lines else "(empty)"


def _readFileContent(path: str, max_chars: int = 5000) -> str:
    """Read a file's content (read-only)."""
    home = os.path.expanduser("~")
    resolved = _resolveObservePath(path)
    if not resolved or not os.path.isfile(resolved):
        return f"Error: Cannot read {path} - invalid or not a file"

    # Security: ensure resolved path is under home
    if not resolved.startswith(home):
        return f"Error: Access denied - {path} is outside allowed area"

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1000)
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n...(truncated at {max_chars} chars)"
        return content
    except PermissionError:
        return f"Error: Permission denied"
    except Exception as e:
        return f"Error reading file: {e}"


def observe(path: str, mode: str = "tree", max_items: int = 50) -> str:
    """Observe a directory outside the sandbox (read-only).

    Args:
        path: Path to observe (supports ~ and relative to home)
        mode: "tree" (directory listing), "read" (read files), "report" (generate summary)
        max_items: Max items to show in tree mode

    Examples:
        observe("~/apps") - list what's in ~/apps
        observe("~/apps/pixelworld", mode="tree") - tree view of pixelworld
        observe("~/notes", mode="report") - generate a summary report
    """
    home = os.path.expanduser("~")
    resolved = _resolveObservePath(path)

    if not resolved:
        return "Error: Invalid path. Use ~/path or absolute path under home."

    if not resolved.startswith(home):
        return f"Error: Access denied. Only paths under ~ are allowed."

    if mode == "report":
        # Generate a summary report
        if not os.path.isdir(resolved):
            return f"Error: {path} is not a directory"
        report_lines = [f"# Observation Report: {path}", ""]
        report_lines.append(f"**Path:** {resolved}")
        report_lines.append(f"**Scanned:** {datetime.now().isoformat()}", "")

        try:
            entries = os.listdir(resolved)
            dirs = [e for e in entries if os.path.isdir(os.path.join(resolved, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(resolved, e))]

            report_lines.append(
                f"**Summary:** {len(dirs)} directories, {len(files)} files"
            )
            report_lines.append("")

            if dirs:
                report_lines.append("## Directories")
                for d in sorted(dirs)[:20]:
                    subpath = os.path.join(resolved, d)
                    try:
                        subcount = len(os.listdir(subpath))
                        report_lines.append(f"- 📁 {d}/ ({subcount} items)")
                    except:
                        report_lines.append(f"- 📁 {d}/")
                if len(dirs) > 20:
                    report_lines.append(f"... and {len(dirs) - 20} more")

            if files:
                report_lines.append("")
                report_lines.append("## Files")
                for f in sorted(files)[:20]:
                    fpath = os.path.join(resolved, f)
                    try:
                        size = os.path.getsize(fpath)
                        if size > 1024 * 1024:
                            size_str = f"{size / (1024 * 1024):.1f}MB"
                        elif size > 1024:
                            size_str = f"{size / 1024:.1f}KB"
                        else:
                            size_str = f"{size}B"
                        report_lines.append(f"- 📄 {f} ({size_str})")
                    except:
                        report_lines.append(f"- 📄 {f}")
                if len(files) > 20:
                    report_lines.append(f"... and {len(files) - 20} more")

        except Exception as e:
            report_lines.append(f"Error scanning: {e}")

        return "\n".join(report_lines)

    elif mode == "read":
        # Read file content
        if os.path.isfile(resolved):
            return _readFileContent(resolved)
        elif os.path.isdir(resolved):
            # If it's a dir, show tree instead
            return _getDirectoryTree(resolved, max_items=max_items)
        else:
            return f"Error: {path} not found"

    else:  # tree mode
        return _getDirectoryTree(resolved, max_items=max_items)


def execute_tool(name: str, arguments: dict, env_root: str) -> str:
    """Run a tool by name."""
    if name == "shell":
        return run_command(arguments["command"], env_root)
    elif name == "fetch_url":
        return fetch_url(arguments.get("url", ""))
    elif name == "web_search":
        return ollama_web_search(
            arguments.get("query", ""),
            arguments.get("max_results", 5),
        )
    elif name == "web_fetch":
        return ollama_web_fetch(arguments.get("url", ""))
    elif name == "observe":
        return observe(
            arguments.get("path", ""),
            arguments.get("mode", "tree"),
            arguments.get("max_items", 50),
        )
    else:
        return f"Unknown tool: {name}"
