import importlib.metadata

from contrastset import __version__, cli


def test_version_matches_package_metadata_and_cli(capsys) -> None:  # type: ignore[no-untyped-def]
    assert __version__ == importlib.metadata.version("contrastset")
    try:
        cli.main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert capsys.readouterr().out.strip() == f"contrastset {__version__}"
