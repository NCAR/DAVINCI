"""Derived-analysis package: EOF, wavelet, and the shared base/registry.

Importing this package registers all concrete analyses as an import
side-effect (added in later plans). The registry itself lives in
``davinci_monet.core.registry`` to avoid circular imports.
"""

from __future__ import annotations

from davinci_monet.analysis import aod_preprocess as _aod_preprocess  # noqa: F401
from davinci_monet.analysis import eof as _eof  # noqa: F401  (registers "eof")
from davinci_monet.analysis import gridded as _gridded  # noqa: F401
from davinci_monet.analysis import known_truth as _known_truth  # noqa: F401
from davinci_monet.analysis import mmr_writer as _mmr_writer  # noqa: F401
from davinci_monet.analysis import projection as _projection  # noqa: F401
from davinci_monet.analysis import scaling as _scaling  # noqa: F401
from davinci_monet.analysis import wavelet as _wavelet  # noqa: F401  (registers "wavelet")
from davinci_monet.analysis import wavelet_filter as _wavelet_filter  # noqa: F401
from davinci_monet.analysis.base import (
    AnalysisResult,
    AnalysisRuntime,
    ArtifactDeclaration,
    DerivedAnalysis,
)
from davinci_monet.analysis.cwt_core import CWTResult, cwt_reconstruct, cwt_transform

__all__ = [
    "AnalysisResult",
    "AnalysisRuntime",
    "ArtifactDeclaration",
    "CWTResult",
    "DerivedAnalysis",
    "cwt_reconstruct",
    "cwt_transform",
]
