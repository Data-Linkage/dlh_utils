coverage run -m pytest .
coverage report -m | grep -v local | grep -v cloudera | grep -v __ | grep -v conftest | grep -v tests | grep -v _version
