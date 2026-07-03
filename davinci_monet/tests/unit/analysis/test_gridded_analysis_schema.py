from __future__ import annotations

import pytest

from davinci_monet.config.schema import MonetConfig


def _base_sources() -> dict:
    return {
        "cam": {
            "type": "generic",
            "files": "cam.nc",
            "variables": {
                "AODNDG_OBS": {},
                "AODNDG_MODEL_PRE": {},
                "AODNDG_MODEL_POST": {},
                "AODNDG_MASK": {},
            },
        }
    }


def test_gridded_analysis_spec_parses_roles_fields_and_groupby() -> None:
    cfg = MonetConfig.model_validate(
        {
            "sources": _base_sources(),
            "analyses": {
                "daily_aod": {
                    "type": "gridded_analysis",
                    "source": "cam",
                    "groupby": "day",
                    "roles": {
                        "observation": "AODNDG_OBS",
                        "first_guess": "AODNDG_MODEL_PRE",
                        "analysis": "AODNDG_MODEL_POST",
                        "mask": "AODNDG_MASK",
                    },
                    "fields": {
                        "analyzed_aod": {"formula": 'mean(analysis, dim="time")'},
                        "nudge_fraction": {"formula": 'mean(mask, dim="time")'},
                    },
                }
            },
        }
    )
    spec = cfg.analyses["daily_aod"]
    assert spec.type == "gridded_analysis"
    assert spec.source == "cam"
    assert spec.groupby == "day"
    assert spec.roles["analysis"] == "AODNDG_MODEL_POST"
    assert spec.fields["analyzed_aod"].formula == 'mean(analysis, dim="time")'


def test_gridded_analysis_requires_at_least_one_role_and_field() -> None:
    with pytest.raises(ValueError, match="roles"):
        MonetConfig.model_validate(
            {
                "sources": _base_sources(),
                "analyses": {
                    "bad": {
                        "type": "gridded_analysis",
                        "source": "cam",
                        "groupby": "day",
                        "roles": {},
                        "fields": {"x": {"formula": "1"}},
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="fields"):
        MonetConfig.model_validate(
            {
                "sources": _base_sources(),
                "analyses": {
                    "bad": {
                        "type": "gridded_analysis",
                        "source": "cam",
                        "groupby": "day",
                        "roles": {"analysis": "AODNDG_MODEL_POST"},
                        "fields": {},
                    }
                },
            }
        )
