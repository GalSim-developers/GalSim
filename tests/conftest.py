def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="Run slow tests"
    )
