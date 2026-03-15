from rich.console import Console
from rich.theme import Theme

_theme = Theme(
    {
        "info": "cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "muted": "dim",
    }
)

console = Console(theme=_theme)
