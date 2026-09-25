import pytest

from monolith.core import load_profiles, profiles_dir
from monolith.core.profile import Profile


def test_repo_profiles_load():
    ps = load_profiles()
    assert {"apple-m3-pro-18c", "apple-m5-pro-20c"} <= set(ps)
    m5 = ps["apple-m5-pro-20c"]
    assert m5.family == "Apple10" and m5.key == "apple10" and m5.gpu_cores == 20 and m5.lane_order == "interleaved16"
    assert m5.sibling_order == "alu_first" and m5.max_cb_ms == 16
    m3 = ps["apple-m3-pro-18c"]
    assert m3.key == "apple9" and m3.lane_order == "contiguous" and (profiles_dir() / "README.md").exists()


def test_cost_tables_measured_points_and_interpolation():
    m5 = load_profiles()["apple-m5-pro-20c"]
    assert m5.cost("fp8", 1) == 1.0 and m5.cost("fp8", 2) == pytest.approx(1.08) and m5.cost("nvfp4", 4) == pytest.approx(1.79)
    assert m5.cost("fp8", 3) == pytest.approx((1.08 + 1.11) / 2)         # linear between T = 2 and T = 4
    assert m5.cost("accelerator_fp8", 8) == pytest.approx(1.09) and m5.cost("accelerator_nvfp4", 16) == pytest.approx(1.04)   # gemm_tile (#50)
    with pytest.raises(ValueError):
        m5.cost("fp8", 9)                                                    # never extrapolate
    with pytest.raises(KeyError):
        load_profiles()["apple-m3-pro-18c"].cost("fp8", 2)                  # unmeasured there


def test_profile_validation():
    with pytest.raises(ValueError):
        Profile.from_dict("x", {"gpu_cores": 1, "nominal_gbps": 1.0})                      # no engine block
    with pytest.raises(ValueError):
        Profile.from_dict("x", {"gpu_cores": 1, "nominal_gbps": 1.0, "engine": {"family": "Apple9", "lane_order": "zigzag"}})
    with pytest.raises(ValueError):
        Profile.from_dict("x", {"gpu_cores": 1, "nominal_gbps": 1.0,
                                "engine": {"family": "Apple9", "lane_order": "contiguous", "cost_T": {"fp8": {"1": 1.2}}}})


def test_accelerator_fields():
    from monolith.core.profile import Profile, load_profiles

    base = {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}}
    assert Profile.from_dict("p", base).accelerator == "off" and Profile.from_dict("p", base).accelerator_min_t == {}
    on = Profile.from_dict("p", dict(base, engine=dict(base["engine"], accelerator="on", accelerator_min_t={"nvfp4": 2, "fp8": 4})))
    assert on.accelerator == "on" and on.accelerator_min_t == {"nvfp4": 2, "fp8": 4}
    with pytest.raises(ValueError):
        Profile.from_dict("p", dict(base, engine=dict(base["engine"], accelerator="maybe")))
    with pytest.raises(ValueError):
        Profile.from_dict("p", dict(base, engine=dict(base["engine"], accelerator_min_t={"nvfp4": 0})))
    m5 = load_profiles()["apple-m5-pro-20c"]
    assert m5.accelerator == "on" and m5.accelerator_min_t["nvfp4"] == 2 and m5.cost("accelerator_nvfp4", 8) == pytest.approx(1.03)
