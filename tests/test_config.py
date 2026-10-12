import subprocess
import sys

from pafbench import config
from pafbench.config import CELLS_TUNED_SETTINGS, CHOSEN_SETTINGS, IMPACT_SETTINGS, OUT, REPO
from pafbench.flags import Setting, setting_tag
from pafbench.grid import all_settings, grid, published


def test_impact_settings_order():
    # the position of a tag is its bit in stage 23's removal mask
    assert [setting_tag(s) for s in IMPACT_SETTINGS] == [
        "c250_l7_m1_e", "c200_l10_m2_i",
        "c300_l10_m1_e", "c300_l10_m1_i", "c275_l9_m1_e", "c275_l9_m1_i",
        "c180_l11_m2_e", "c180_l11_m2_i", "c375_l3_m2_e", "c375_l3_m2_i",
        "c250_l7_m2_e", "c450_l5_m1_i",
        "c250_lnone_m0_e", "c200_lnone_m0_i",
    ]


def test_impact_settings_are_grid_published_or_control():
    assert len(IMPACT_SETTINGS) == 14 and len({setting_tag(s) for s in IMPACT_SETTINGS}) == 14
    assert all(isinstance(s, Setting) for s in IMPACT_SETTINGS)
    assert set(IMPACT_SETTINGS) <= set(all_settings())
    assert IMPACT_SETTINGS[:2] == CHOSEN_SETTINGS and IMPACT_SETTINGS[2:10] == tuple(published())
    assert IMPACT_SETTINGS[10:12] == CELLS_TUNED_SETTINGS
    candidates = {s for s in grid() if s.min_len is not None}
    assert set(CHOSEN_SETTINGS) | set(CELLS_TUNED_SETTINGS) <= candidates


def test_controls_share_covlen_and_mode_with_the_chosen_settings():
    controls = IMPACT_SETTINGS[12:]
    assert [s for s in IMPACT_SETTINGS if s.min_len is None] == list(controls)
    assert set(controls) <= {s for s in grid() if s.min_len is None}
    for chosen, control in zip(CHOSEN_SETTINGS, controls):
        assert control == Setting(chosen.covlen, None, 0, chosen.introns)
    assert [s.introns for s in CHOSEN_SETTINGS] == [False, True]


def test_thresholds():
    want = {
        "XF_RECOUNT_TOLERANCE": 0, "RULE_AGREEMENT_MAX": 1e-6, "A4_PARALLEL": 4, "UNIT_BIN": 1000, "CLASS_MARGIN": 0.1,
        "KNN_K": 15, "KNN_FOLDS": 5, "THIN_REPLICATES": 1, "THIN_REPLICATES_CHOSEN": 3, "MIN_CELLS_GENE": 3, "RARE_PCTL": 99,
        "TOP_GENES": 20, "MARKER_LOSS": 0.5, "DOWNSAMPLE_DRAWS": 100, "JACCARD_DISSOLVED": 0.5, "JACCARD_STABLE": 0.75,
        "BIAS_POINTS": 5, "EXPR_BINS": 4, "MIN_STRATUM": 50, "POOL_MIN": 30, "REGION_REF_MIN": 200, "SITE_TOL": 5,
        "LONG_END_TOL": 10, "NEAR_INTRON_TOL": 5, "MIN_INTRON": 50, "FIVE_PRIME_TOL": 50, "NULL_SHIFT": 1000,
        "DIST_END_BINS": (100, 300, 1000, 5000), "RERUN_DIFF": 0.001,
    }
    for name, value in want.items():
        assert getattr(config, name) == value, name
    assert config.CLUSTER == {"target_sum": 10_000, "n_hvg": 2000, "n_pcs": 30, "k": 15, "resolution": 1.0,
                              "resolutions_alt": (0.5, 2.0), "seeds": tuple(range(10))}
    assert not hasattr(config, "RULE_VERSION")


def test_paths():
    assert config.REPORT_DIR == OUT / "report"
    assert str(config.PIGEON_DIR) == "/data/long_read_annotation_tool/test_dataset/input_files"
    assert config.POLYASITE.name == "atlas.clusters.3.0.GRCh38.GENCODE_42.bed.gz"


def test_every_module_imports_first():
    # config builds IMPACT_SETTINGS from flags and grid, and flags reaches vendor: no import order may break that
    for mod in ("config", "flags", "grid", "vendor", "transcripts"):
        subprocess.run([sys.executable, "-B", "-c", f"import pafbench.{mod}"], cwd=REPO, check=True)
