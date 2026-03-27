from __future__ import annotations

from c3rnt2 import autolearn


def test_issue_context_lines_expands_window() -> None:
    lines = autolearn._issue_context_lines({10}, total_lines=20, window=2)
    assert lines == [8, 9, 10, 11, 12]


def test_issue_context_lines_clamps_bounds() -> None:
    lines = autolearn._issue_context_lines({1, 20}, total_lines=20, window=3)
    assert lines[0] == 1
    assert lines[-1] == 20
    assert all(1 <= x <= 20 for x in lines)


def test_parse_and_validate_urls_respects_allowlist() -> None:
    text = """
https://docs.python.org/3/tutorial/
https://owasp.org/www-project-top-ten/
https://example.com/hack
"""
    urls = autolearn._parse_and_validate_urls(
        text,
        allowed_domains=["docs.python.org", "owasp.org"],
    )
    assert "https://docs.python.org/3/tutorial/" in urls
    assert "https://owasp.org/www-project-top-ten/" in urls
    assert all("example.com" not in u for u in urls)


def test_parse_and_validate_urls_blocks_non_https() -> None:
    text = """
http://docs.python.org/3/tutorial/
https://docs.python.org/3/library/
"""
    urls = autolearn._parse_and_validate_urls(text, allowed_domains=["docs.python.org"])
    assert urls == ["https://docs.python.org/3/library/"]
