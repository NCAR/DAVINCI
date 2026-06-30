import pytest

from davinci_monet.datasets.satellite.catalog.registry import (
    Catalog,
    UnknownProductError,
    get_catalog,
)
from davinci_monet.datasets.satellite.catalog.schema import ProductEntry, VariableEntry


def test_variable_entry_minimal():
    v = VariableEntry(
        display_name="aod_550nm", sds_name="Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"
    )
    assert v.display_name == "aod_550nm"
    assert v.sds_name.startswith("Aerosol_Optical_Depth")
    assert v.units == "1"  # default


def test_product_entry_resolves_variable_by_display_and_sds():
    p = ProductEntry(
        product_id="MOD08_M3",
        instrument="MODIS",
        platform="Terra",
        daac="LAADS",
        collection="061",
        level="L3",
        geometry="GRID",
        file_format="HDF4",
        time_parse="A%Y%j",
        dim_aliases={"XDim": "lon", "YDim": "lat"},
        variables=[
            VariableEntry(
                display_name="aod_550nm", sds_name="Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"
            )
        ],
    )
    by_display = p.variable_by_display("aod_550nm")
    assert by_display is not None
    assert by_display.sds_name.startswith("Aerosol_Optical_Depth")
    by_sds = p.variable_by_sds("Aerosol_Optical_Depth_Land_Ocean_Mean_Mean")
    assert by_sds is not None
    assert by_sds.display_name == "aod_550nm"
    assert p.variable_by_display("nope") is None


def test_catalog_resolves_known_products():
    cat = get_catalog()
    terra = cat.resolve("MOD08_M3")
    aqua = cat.resolve("MYD08_M3")
    assert terra.platform == "Terra"
    assert aqua.platform == "Aqua"
    aod = terra.variable_by_display("aod_550nm")
    assert aod is not None
    assert aod.wavelength_nm == 550


def test_catalog_unknown_product_suggests_matches():
    cat = get_catalog()
    with pytest.raises(UnknownProductError) as exc:
        cat.resolve("MOD08_X3")
    assert "MOD08_M3" in str(exc.value)  # close-match suggestion


def test_catalog_resolves_viirs_aerdb_l2():
    cat = get_catalog()
    product = cat.resolve("AERDB_L2_VIIRS_NOAA20")
    assert product.level == "L2"
    assert product.geometry == "SWATH"
    assert product.instrument == "VIIRS"
    assert product.platform == "NOAA-20"
    land = product.variable_by_display("aod_550_land")
    assert land is not None
    assert land.wavelength_nm == 550
    assert cat.resolve("viirs_aerdb_noaa20").product_id == "AERDB_L2_VIIRS_NOAA20"
