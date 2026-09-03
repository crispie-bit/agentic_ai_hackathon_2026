"""Service layer for external integrations and infrastructure stubs."""

from agentic_system.services.azure_setup import azure_status, build_graph_auth_url
from agentic_system.services.course_memory import CourseMemoryStore
from agentic_system.services.course_retriever import CourseRetriever
from agentic_system.services.document_ingestion import DocumentIngestionService
from agentic_system.services.ntulearn_browser import NTULearnBrowser

__all__ = [
    "CourseMemoryStore",
    "CourseRetriever",
    "DocumentIngestionService",
    "NTULearnBrowser",
    "azure_status",
    "build_graph_auth_url",
]
