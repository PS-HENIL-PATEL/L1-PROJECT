"""
Enterprise RAG OS — Root Package
=================================

Purpose:
    Root package marker for the Enterprise RAG OS application.
    Exposes the application version for use across the codebase.

Architecture:
    This sits at the top of the application package hierarchy.
    All sub-packages (config, api, core, etc.) are children of this package.
"""

__version__ = "0.1.0"
__app_name__ = "Enterprise RAG OS"
