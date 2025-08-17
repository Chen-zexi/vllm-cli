#!/usr/bin/env python3
"""
Log viewer module for vLLM CLI.

Provides interactive log viewing functionality with scrolling, search,
and real-time tailing capabilities.
"""
import logging
import os
import subprocess
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import List, Optional, Tuple

from rich.text import Text

from ..server import VLLMServer
from .common import console, create_panel
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


class LogViewer:
    """
    Interactive log viewer with scrolling and search capabilities.
    """

    def __init__(self, log_path: str):
        """
        Initialize the log viewer.

        Args:
            log_path: Path to the log file to view
        """
        self.log_path = Path(log_path)
        self.lines: List[str] = []
        self.current_line = 0
        self.search_term = ""
        self.search_results: List[int] = []
        self.search_index = 0
        self.tail_mode = False
        self.tail_thread: Optional[Thread] = None
        self.tail_stop_event = Event()
        self.tail_queue: Queue = Queue()

    def load_log_file(self) -> bool:
        """
        Load the log file contents.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.log_path.exists():
                return False

            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                self.lines = [line.rstrip("\n\r") for line in f.readlines()]
            return True
        except Exception as e:
            logger.error(f"Failed to load log file {self.log_path}: {e}")
            return False

    def search_logs(self, term: str) -> List[int]:
        """
        Search for a term in the log lines.

        Args:
            term: Search term

        Returns:
            List of line numbers containing the term
        """
        self.search_term = term.lower()
        self.search_results = []

        for i, line in enumerate(self.lines):
            if self.search_term in line.lower():
                self.search_results.append(i)

        self.search_index = 0
        return self.search_results

    def jump_to_next_match(self) -> bool:
        """
        Jump to next search match.

        Returns:
            True if jumped, False if no more matches
        """
        if not self.search_results:
            return False

        if self.search_index < len(self.search_results) - 1:
            self.search_index += 1
        else:
            self.search_index = 0  # Wrap to first match

        self.current_line = self.search_results[self.search_index]
        return True

    def jump_to_prev_match(self) -> bool:
        """
        Jump to previous search match.

        Returns:
            True if jumped, False if no matches
        """
        if not self.search_results:
            return False

        if self.search_index > 0:
            self.search_index -= 1
        else:
            self.search_index = len(self.search_results) - 1  # Wrap to last match

        self.current_line = self.search_results[self.search_index]
        return True

    def get_visible_lines(self, height: int) -> Tuple[List[str], int, int]:
        """
        Get lines visible in the current viewport.

        Args:
            height: Number of lines to show

        Returns:
            Tuple of (visible lines, start line, end line)
        """
        if not self.lines:
            return [], 0, 0

        # Ensure current_line is within bounds
        self.current_line = max(0, min(self.current_line, len(self.lines) - 1))

        # Calculate viewport
        start_line = max(0, self.current_line - height // 2)
        end_line = min(len(self.lines), start_line + height)

        # Adjust start_line if we're near the end
        if end_line - start_line < height and start_line > 0:
            start_line = max(0, end_line - height)

        visible_lines = []
        for i in range(start_line, end_line):
            line = self.lines[i]
            line_num = f"{i + 1:6d}"

            # Highlight current line
            if i == self.current_line:
                line_style = "reverse"
            # Highlight search matches
            elif self.search_term and self.search_term in line.lower():
                line_style = "yellow"
            else:
                line_style = "white"

            # Format line with number
            formatted_line = (
                f"[cyan]{line_num}[/cyan] [{line_style}]{line}[/{line_style}]"
            )
            visible_lines.append(formatted_line)

        return visible_lines, start_line, end_line

    def start_tail_mode(self) -> None:
        """Start tailing the log file in real-time."""
        if self.tail_mode or not self.log_path.exists():
            return

        self.tail_mode = True
        self.tail_stop_event.clear()
        self.tail_thread = Thread(target=self._tail_worker, daemon=True)
        self.tail_thread.start()

    def stop_tail_mode(self) -> None:
        """Stop tailing the log file."""
        if not self.tail_mode:
            return

        self.tail_mode = False
        self.tail_stop_event.set()
        if self.tail_thread:
            self.tail_thread.join(timeout=1)

    def _tail_worker(self) -> None:
        """Worker thread for tailing the log file."""
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                # Seek to end
                f.seek(0, 2)

                while not self.tail_stop_event.is_set():
                    line = f.readline()
                    if line:
                        self.tail_queue.put(line.rstrip("\n\r"))
                    else:
                        time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in tail worker: {e}")

    def update_from_tail(self) -> bool:
        """
        Update lines from tail queue.

        Returns:
            True if new lines were added
        """
        if not self.tail_mode:
            return False

        new_lines = []
        try:
            while True:
                line = self.tail_queue.get_nowait()
                new_lines.append(line)
        except Empty:
            pass

        if new_lines:
            self.lines.extend(new_lines)
            # Auto-scroll to end in tail mode
            self.current_line = len(self.lines) - 1
            return True

        return False


def display_full_log(log_path: str) -> str:
    """
    Display the complete log file content.

    Args:
        log_path: Path to the log file

    Returns:
        Action taken by user
    """
    if not Path(log_path).exists():
        console.print(f"[red]Log file not found: {log_path}[/red]")
        return "back"

    try:
        console.clear()
        console.print(f"[bold cyan]Full Log - {Path(log_path).name}[/bold cyan]")
        console.print("[dim]" + "=" * 80 + "[/dim]")

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if content.strip():
                console.print(content)
            else:
                console.print("[dim]Log file is empty[/dim]")

        console.print("\n[dim]" + "=" * 80 + "[/dim]")
        console.print(f"[dim]Log file: {log_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error reading log file: {e}[/red]")

    input("\nPress Enter to continue...")
    return "back"


def display_log_tail(log_path: str, lines: int = 50) -> str:
    """
    Display the last N lines of the log file.

    Args:
        log_path: Path to the log file
        lines: Number of lines to show from the end

    Returns:
        Action taken by user
    """
    if not Path(log_path).exists():
        console.print(f"[red]Log file not found: {log_path}[/red]")
        return "back"

    try:
        console.clear()
        console.print(
            f"[bold cyan]Last {lines} Lines - {Path(log_path).name}[/bold cyan]"
        )
        console.print("[dim]" + "=" * 80 + "[/dim]")

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

            if all_lines:
                tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                for line in tail_lines:
                    console.print(line.rstrip("\n\r"))
            else:
                console.print("[dim]Log file is empty[/dim]")

        console.print("\n[dim]" + "=" * 80 + "[/dim]")
        console.print(f"[dim]Showing last {lines} lines of {log_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error reading log file: {e}[/red]")

    input("\nPress Enter to continue...")
    return "back"


def view_full_log(log_path: str) -> str:
    """
    Display an interactive log viewer.

    Args:
        log_path: Path to the log file

    Returns:
        Action taken by user
    """
    if not Path(log_path).exists():
        console.print(f"[red]Log file not found: {log_path}[/red]")
        return "back"

    viewer = LogViewer(log_path)
    if not viewer.load_log_file():
        console.print(f"[red]Failed to load log file: {log_path}[/red]")
        return "back"

    console.clear()

    # Calculate available height for log content
    console_height = console.height
    header_height = 4  # Title + instructions
    footer_height = 3  # Status + controls
    log_height = max(10, console_height - header_height - footer_height)

    # Instructions
    instructions = (
        "[cyan]Controls:[/cyan] ↑/↓ Navigate | PgUp/PgDn Fast scroll | "
        "Home/End Jump to start/end | S Search | T Tail mode | O Open in editor | Q Quit"
    )

    try:
        while True:
            # Get visible content
            visible_lines, start_line, end_line = viewer.get_visible_lines(log_height)

            if not visible_lines:
                content = Text("Log file is empty", style="dim yellow")
            else:
                content = Text("\n".join(visible_lines))

            # Create status info
            status_info = []
            total_lines = len(viewer.lines)
            current_pos = viewer.current_line + 1

            status_info.append(f"Line {current_pos}/{total_lines}")

            if viewer.search_term:
                match_info = f"Search: '{viewer.search_term}' ({len(viewer.search_results)} matches)"
                if viewer.search_results:
                    match_info += (
                        f" [{viewer.search_index + 1}/{len(viewer.search_results)}]"
                    )
                status_info.append(match_info)

            if viewer.tail_mode:
                status_info.append("[green]TAIL MODE[/green]")

            status_text = " | ".join(status_info)

            # Create panels
            header_panel = create_panel(
                instructions,
                title=f"Log Viewer - {Path(log_path).name}",
                border_style="cyan",
            )

            content_panel = create_panel(
                content, title="Log Content", border_style="white"
            )

            footer_panel = create_panel(status_text, title="Status", border_style="dim")

            # Display
            console.clear()
            console.print(header_panel)
            console.print(content_panel)
            console.print(footer_panel)

            # Handle input (simplified for now - in real implementation would use keyboard input)
            console.print("\n[dim]Available actions:[/dim]")
            actions = [
                "↑ Up",
                "↓ Down",
                "Search",
                "Tail Mode",
                "Open in Editor",
                "← Back",
            ]

            if viewer.tail_mode:
                actions = ["Stop Tail"] + actions[:-1] + ["← Back"]

            action = unified_prompt(
                "log_action", "Choose action", actions, allow_back=False
            )

            if not action or action == "← Back":
                viewer.stop_tail_mode()
                return "back"
            elif action == "↑ Up":
                viewer.current_line = max(0, viewer.current_line - 1)
            elif action == "↓ Down":
                viewer.current_line = min(
                    len(viewer.lines) - 1, viewer.current_line + 1
                )
            elif action == "Search":
                search_term = input("Enter search term: ").strip()
                if search_term:
                    matches = viewer.search_logs(search_term)
                    if matches:
                        console.print(f"[green]Found {len(matches)} matches[/green]")
                        viewer.current_line = matches[0]
                    else:
                        console.print("[yellow]No matches found[/yellow]")
                        time.sleep(1)
            elif action == "Tail Mode":
                viewer.start_tail_mode()
                console.print(
                    "[green]Tail mode started. Watching for new log entries...[/green]"
                )
                time.sleep(1)
            elif action == "Stop Tail":
                viewer.stop_tail_mode()
                console.print("[yellow]Tail mode stopped[/yellow]")
                time.sleep(1)
            elif action == "Open in Editor":
                try:
                    editor = os.environ.get("EDITOR", "nano")
                    subprocess.run([editor, str(log_path)], check=True)
                except Exception as e:
                    console.print(f"[red]Failed to open editor: {e}[/red]")
                    time.sleep(2)

            # Update from tail if active
            if viewer.tail_mode:
                viewer.update_from_tail()

    except KeyboardInterrupt:
        viewer.stop_tail_mode()
        return "back"
    except Exception as e:
        logger.error(f"Error in log viewer: {e}")
        viewer.stop_tail_mode()
        console.print(f"[red]Error viewing logs: {e}[/red]")
        return "back"


def show_log_menu(server: VLLMServer) -> str:
    """
    Show log viewing options for a server.

    Args:
        server: VLLMServer instance

    Returns:
        Action taken by user
    """
    if not server.log_path or not server.log_path.exists():
        console.print("[yellow]No log file available for this server[/yellow]")
        return "back"

    options = [
        "Display Full Log",
        "Show Last 50 Lines",
        "Show Last 100 Lines",
        "Open in External Editor",
        "Show Log Path",
    ]

    action = unified_prompt("log_option", f"Log Options - {server.model}", options)

    if not action or action == "← Back":
        return "back"
    elif action == "Display Full Log":
        return display_full_log(str(server.log_path))
    elif action == "Show Last 50 Lines":
        return display_log_tail(str(server.log_path), 50)
    elif action == "Show Last 100 Lines":
        return display_log_tail(str(server.log_path), 100)
    elif action == "Open in External Editor":
        try:
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, str(server.log_path)], check=True)
            return "back"
        except Exception as e:
            console.print(f"[red]Failed to open editor: {e}[/red]")
            return "back"
    elif action == "Show Log Path":
        console.print("\n[cyan]Log file location:[/cyan]")
        console.print(f"[white]{server.log_path}[/white]")
        console.print(
            "\n[dim]You can open this file in any text editor or use 'tail -f' to follow it.[/dim]"
        )
        input("\nPress Enter to continue...")
        return "back"

    return "back"


def select_server_for_logs() -> Optional[VLLMServer]:
    """
    Let user select a server to view logs for.

    Returns:
        Selected VLLMServer or None if cancelled
    """
    from ..server import get_active_servers

    servers = get_active_servers()
    if not servers:
        console.print("[yellow]No active servers found[/yellow]")
        return None

    if len(servers) == 1:
        return servers[0]

    # Create server selection menu
    server_choices = []
    for server in servers:
        status = "Running" if server.is_running() else "Stopped"
        choice = f"{server.model} (Port {server.port}) - {status}"
        server_choices.append(choice)

    selection = unified_prompt(
        "server_select", "Select server to view logs", server_choices
    )

    if not selection or selection == "← Back":
        return None

    # Find selected server
    for i, choice in enumerate(server_choices):
        if choice == selection:
            return servers[i]

    return None
