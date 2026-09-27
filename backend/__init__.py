"""VisionGuard AI backend: API layer and application entry point.

Depends on ``computer_vision``; never the other way round. The vision engine
must stay runnable with no server, no API and no frontend.
"""

from computer_vision import __phase__, __version__

__all__ = ["__phase__", "__version__"]
