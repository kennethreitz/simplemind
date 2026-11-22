import pytest
from pydantic import SecretStr

from simplemind.providers import AtlasCloud, providers
from simplemind.utils import find_provider


def test_atlascloud_defaults():
    provider = AtlasCloud(api_key="test-key")

    assert provider.NAME == "atlascloud"
    assert provider.DEFAULT_MODEL == "deepseek-ai/deepseek-v4-flash"
    assert str(provider.client.base_url).rstrip("/") == "https://api.atlascloud.ai/v1"


def test_atlascloud_registered():
    assert AtlasCloud in providers
    assert isinstance(find_provider("atlascloud"), AtlasCloud)


def test_atlascloud_reads_key_from_env(monkeypatch):
    from simplemind.settings import settings

    monkeypatch.setattr(settings, "ATLASCLOUD_API_KEY", SecretStr("env-key"))
    assert AtlasCloud().api_key == "env-key"


def test_atlascloud_requires_key(monkeypatch):
    from simplemind.settings import settings

    monkeypatch.setattr(settings, "ATLASCLOUD_API_KEY", None)
    with pytest.raises(ValueError):
        AtlasCloud().client
