"""FastAPI dependency providers.

Routers depend on these annotated aliases instead of importing services
directly, so a test can swap any single piece via `dependency_overrides`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.ingestor import DocumentIngestor
from app.protocols import Generator
from app.retrieval import Retriever
from app.services import Services
from app.vector_db import Database


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_settings_dep(
    services: Annotated[Services, Depends(get_services)],
) -> Settings:
    return services.settings


def get_db(services: Annotated[Services, Depends(get_services)]) -> Database:
    return services.db


def get_retriever(services: Annotated[Services, Depends(get_services)]) -> Retriever:
    return services.retriever


def get_generator(services: Annotated[Services, Depends(get_services)]) -> Generator:
    return services.generator


def get_ingestor(
    services: Annotated[Services, Depends(get_services)],
) -> DocumentIngestor:
    return services.ingestor


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbDep = Annotated[Database, Depends(get_db)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
GeneratorDep = Annotated[Generator, Depends(get_generator)]
IngestorDep = Annotated[DocumentIngestor, Depends(get_ingestor)]
