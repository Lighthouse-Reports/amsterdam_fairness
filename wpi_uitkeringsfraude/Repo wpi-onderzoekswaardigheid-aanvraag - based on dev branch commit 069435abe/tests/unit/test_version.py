import re

import wpi_onderzoekswaardigheid_aanvraag


def test_version():
    result = wpi_onderzoekswaardigheid_aanvraag.__version__
    print(f"Version is: {result}")
    simplified_regex = r"\d+\.\d+\.\d+([a-zA-Z]?\d+)?"
    assert re.match(simplified_regex, result)
