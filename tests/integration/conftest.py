import pytest
import sys
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests import Session


pytest_plugin = ["docker_compose"]


@pytest.fixture(name="homepage")
def fixture_homepage(function_scoped_container_getter) -> str:

    service = function_scoped_container_getter.get('web').network_info[0]

    if service.hostname == '0.0.0.0':
        base_url = f"http://localhost:{service.host_port}"
    else:
        base_url = f"http://{service.hostname}:{service.host_port}"

    print("Info service :", base_url, file=sys.stderr)

    retry = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
    )

    session = Session()
    session.mount('http://', HTTPAdapter(max_retries=retry))

    assert session.get(f'{base_url}/health_check').text == 'ok'

    return base_url
