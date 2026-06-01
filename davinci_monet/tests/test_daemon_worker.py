"""Unit tests for the daemon worker (config injection + progress; mocked run)."""

from __future__ import annotations

from davinci_monet.daemon import worker


def test_inject_new_files_overrides_target_source_files() -> None:
    config = {
        "analysis": {"output_dir": "/out"},
        "sources": {
            "modis": {"type": "modis", "files": "/data/modis/*.hdf"},
            "cam": {"type": "cesm_fv", "files": "/data/cam/*.nc"},
        },
    }
    new_files = ["/data/modis/new_b.hdf", "/data/modis/new_a.hdf"]

    out = worker.inject_new_files(config, inject_into="modis", new_files=new_files)

    # Target source files: replaced by the injected list (sorted), filename cleared
    assert out["sources"]["modis"]["files"] == [
        "/data/modis/new_a.hdf",
        "/data/modis/new_b.hdf",
    ]
    assert out["sources"]["modis"].get("filename") is None
    # Other sources untouched
    assert out["sources"]["cam"]["files"] == "/data/cam/*.nc"
    # Original config not mutated in place
    assert config["sources"]["modis"]["files"] == "/data/modis/*.hdf"


def test_inject_new_files_unknown_source_raises() -> None:
    config = {"sources": {"cam": {"type": "cesm_fv", "files": "/x/*.nc"}}}
    try:
        worker.inject_new_files(config, inject_into="missing", new_files=["/y/a.nc"])
    except KeyError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown inject_into source")


def test_inject_new_files_noop_when_inject_into_none() -> None:
    config = {"sources": {"cam": {"files": "/x/*.nc"}}}
    out = worker.inject_new_files(config, inject_into=None, new_files=["/y/a.nc"])
    assert out["sources"]["cam"]["files"] == "/x/*.nc"
