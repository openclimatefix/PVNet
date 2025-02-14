from ocf_data_sampler.select.location import Location
from pvnet.utils import SiteLocationLookup
import xarray as xr
import pytest


@pytest.mark.parametrize(
    "lookup_site_id, expected_x, expected_y, expected_id",
    [
        (1, -1.99106, 48.709865, 1),
        (0, 0.30693, 51.509865, 0),
        (2, -1.56106, 56.203865, 2),
    ],
)
def test_site_location_lookup(lookup_site_id, expected_x, expected_y, expected_id):
    # setup
    site_ids = [0, 1, 2]
    longs = [0.30693, -1.99106, -1.56106]
    lats = [51.509865, 48.709865, 56.203865]
    da_long = xr.DataArray(
        data=longs,
        dims="pv_system_id",
        coords=dict(site_id=(["pv_system_id"], site_ids), long=(["pv_system_id"], longs)),
    )
    da_lat = xr.DataArray(
        data=lats,
        dims="pv_system_id",
        coords=dict(site_id=(["pv_system_id"], site_ids), long=(["pv_system_id"], lats)),
    )
    # Actual testing part
    site_lookup = SiteLocationLookup(long=da_long, lat=da_lat)

    # retrieve location of site 1
    site_location: Location = site_lookup(lookup_site_id)

    assert site_location.x == expected_x
    assert site_location.y == expected_y
    assert site_location.id == expected_id
