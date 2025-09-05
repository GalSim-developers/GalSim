import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="Run slow tests"
    )

# cf. https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)
