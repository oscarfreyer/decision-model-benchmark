"""Original v3 settings for historical regression fixtures; never live authorization."""
from common import ROOT, load_config

def legacy_config():
    return load_config(ROOT/'benchmarking/tests/fixtures/legacy_config.toml')
