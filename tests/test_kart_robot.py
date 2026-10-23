import torch

from spec2mpc.robots.kart import KartParams, load_fixed_kart, sample_karts, sample_runtime

RANGED_FIELDS = [
    "mass_kg", "yaw_inertia_kgm2", "wheelbase_m", "cf_Npr", "cr_Npr", "pac_B", "pac_C", "pac_E",
    "mu_nominal", "max_steer_rad", "max_steer_rate_radps", "max_drive_force_N", "max_brake_force_N",
    "max_force_rate_Nps", "actuator_delay_s", "v_max_mps", "width_m", "length_m", "c_rr", "c_drag",
]


def test_samples_within_ranges(kart_cfg, gen, runtime):
    r = sample_karts(512, kart_cfg, runtime, gen)
    for name in RANGED_FIELDS:
        lo, hi = kart_cfg.ranges[name]
        v = getattr(r.params, name)
        assert v.shape == (512,) and (v >= lo).all() and (v <= hi).all()
    assert torch.allclose(r.params.lf_m + r.params.lr_m, r.params.wheelbase_m)
    assert torch.allclose(r.params.a_lat_max_mps2, r.params.mu_nominal * kart_cfg.gravity_mps2)
    assert r.perception.has_prediction.dtype == torch.bool
    assert (r.perception.range_m >= 10).all() and (r.perception.range_m <= 40).all()


def test_same_seed_same_robots(kart_cfg, runtime):
    a = sample_karts(64, kart_cfg, runtime, torch.Generator().manual_seed(3))
    b = sample_karts(64, kart_cfg, runtime, torch.Generator().manual_seed(3))
    assert torch.equal(a.params.mass_kg, b.params.mass_kg) and a.robot_id == b.robot_id


def test_index_and_cat_roundtrip(kart_cfg, gen, runtime):
    r = sample_karts(10, kart_cfg, runtime, gen)
    parts = [r.params.index(slice(0, 4)), r.params.index(slice(4, 10))]
    assert torch.equal(KartParams.cat(parts).mass_kg, r.params.mass_kg)
    sub = r.index(torch.tensor([2, 5]))
    assert sub.robot_id == [r.robot_id[2], r.robot_id[5]] and sub.params.batch_size == 2
    pp = type(r.perception).cat([r.perception.index(slice(0, 3)), r.perception.index(slice(3, 10))])
    assert pp.has_prediction.dtype == torch.bool and pp.batch_size == 10


def test_derived(kart_cfg, gen, runtime):
    p = sample_karts(8, kart_cfg, runtime, gen).params
    assert (p.radius_m > 0).all() and (p.min_turn_radius_m > 0).all()


def test_runtime_sample(kart_cfg, gen):
    rt = sample_runtime(kart_cfg, gen)
    assert rt.control_rate_hz in (10, 20, 25, 50) and rt.solver_iters in (5, 10, 20, 40)


def test_fixed_evc_loads(kart_cfg, runtime):
    r = load_fixed_kart(kart_cfg, "evc", runtime)
    assert r.params.batch_size == 1 and r.perception.batch_size == 1


def test_successive_calls_distinct_ids(kart_cfg, gen, runtime):
    a = sample_karts(16, kart_cfg, runtime, gen)
    b = sample_karts(16, kart_cfg, runtime, gen)
    assert not set(a.robot_id) & set(b.robot_id)


def test_robot_index_bool_mask(kart_cfg, gen, runtime):
    r = sample_karts(6, kart_cfg, runtime, gen)
    mask = torch.tensor([True, False, True, False, False, True])
    sub = r.index(mask)
    assert sub.robot_id == [r.robot_id[i] for i in (0, 2, 5)]
    assert torch.equal(sub.params.mass_kg, r.params.mass_kg[mask])
