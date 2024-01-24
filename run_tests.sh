cd ~/dlh_utils
pip3 uninstall dlh_utils
pip3 install .
cd ~/dlh_utils/dlh_utils/tests
coverage run -m pytest .
coverage report -m | grep -v local | grep -v cloudera | grep -v __ | grep -v conftest | grep -v tests | grep -v _version
