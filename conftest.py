import pytest


@pytest.fixture(autouse=True)
def _fresh_camera_clock(request):
    """The camera-health tests depend on how old a file is, so reset the ages right before each of those tests."""
    if request.module.__name__ in ("test_e2e", "test_e2e_features"):
        import test_e2e
        test_e2e.refresh_live_files()
    yield
