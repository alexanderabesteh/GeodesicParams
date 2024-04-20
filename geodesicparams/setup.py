from setuptools import find_packages, setup

setup(name = "geodesicparams",
      version = "0.0",
      package_dir = {"": "src"},
      packages = find_packages(where = "src")
      )
