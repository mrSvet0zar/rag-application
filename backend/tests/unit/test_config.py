from app.config import Settings


def test_cors_origins_parsing():
    s = Settings(cors_origins="http://a.com, http://b.com ,")
    assert s.cors_origins_list == ["http://a.com", "http://b.com"]


def test_demo_mode_toggle():
    assert Settings(anthropic_api_key="").demo_mode is True
    assert Settings(anthropic_api_key="   ").demo_mode is True
    assert Settings(anthropic_api_key="sk-ant-xyz").demo_mode is False
