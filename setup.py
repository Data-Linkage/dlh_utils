from setuptools import setup

with open("requirements.txt") as f:
	requirements = f.read().splitlines()

setup(
	name="dlh_utils",
	version="1.0",
	description="",
	author="",
	packages=[],
	zip_safe=False,
	install_requirements=requirements
)
