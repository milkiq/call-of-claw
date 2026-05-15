"""Generic TRPG GM agent runtime."""

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
    category=LangChainPendingDeprecationWarning,
)

__all__ = ["__version__"]

__version__ = "0.1.0"
