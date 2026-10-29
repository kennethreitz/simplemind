import pytest

from simplemind.providers import MiniMax, MiniMaxAnthropic, providers
from simplemind.utils import find_provider


@pytest.mark.parametrize(
    ("provider_cls", "protocol", "region", "expected_url"),
    [
        (MiniMax, "openai", "global_en", "https://api.minimax.io/v1"),
        (MiniMax, "openai", "cn_zh", "https://api.minimaxi.com/v1"),
        (
            MiniMaxAnthropic,
            "anthropic",
            "global_en",
            "https://api.minimax.io/anthropic",
        ),
        (
            MiniMaxAnthropic,
            "anthropic",
            "cn_zh",
            "https://api.minimaxi.com/anthropic",
        ),
    ],
)
def test_minimax_endpoints(provider_cls, protocol, region, expected_url):
    provider = provider_cls(api_key="test-key", region=region)

    assert provider.MODEL_IDS == ("MiniMax-M3.1-Flash", "MiniMax-M3")
    assert provider.DEFAULT_MODEL == "MiniMax-M3.1-Flash"
    assert provider.ENDPOINTS[region][protocol] == expected_url


@pytest.mark.parametrize("provider_cls", [MiniMax, MiniMaxAnthropic])
def test_minimax_registered(provider_cls):
    assert provider_cls in providers
    assert isinstance(find_provider(provider_cls.NAME), provider_cls)


def test_minimax_rejects_unknown_region():
    with pytest.raises(ValueError, match="MiniMax API region must be one of"):
        MiniMax(api_key="test-key", region="unknown")
