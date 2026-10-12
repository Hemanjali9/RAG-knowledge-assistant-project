from streamlit.testing.v1 import AppTest


def test_app_displays_main_title():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()

    assert not at.exception
    assert any(
    "Team Knowledge Assistant" in title.value
    for title in at.title
)


def test_app_displays_page_configuration():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()

    assert not at.exception


def test_app_displays_chat_input():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()

    assert not at.exception
    assert len(at.chat_input) > 0