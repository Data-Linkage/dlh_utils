# DLH_utils

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![PyPI version](https://badge.fury.io/py/dlh_utils.svg)](https://badge.fury.io/py/dlh_utils)
[![PyPi Python Versions](https://img.shields.io/pypi/pyversions/dlh-utils.svg)](https://pypi.python.org/pypi/dlh-utils/)

A Python package produced by the Linkage Development team from the Data Linkage Hub at Office for National Statistics (ONS) containing a set of functions used to expedite and streamline the data linkage process.

It's key features include:
* it's scalability to large datasets, using `spark` as a big-data backend
* profiling and flagging functions used to describe and highlight issues in data
* standardisation and cleaning functions to make data comparable ahead of linkage
* linkage functions to derive linkage variables and join data together efficiently

Please log an issue on the issue board or contact any of the active contributors with any issues or suggestions for improvements you have.

## Installation steps
DLH_utils supports Python 3.6+. To install the latest version, simply run:
```sh
pip install dlh_utils
```
Or, if using CDSW, in a terminal session run:
```sh
pip3 install dlh_utils
```

## Demo
For a worked demonstration notebook of these functions being applied within a data linkage context, head over to our [separate demo repository](https://github.com/anthonye93/dlh_utils_demo)

## Common issues

### When using the jaro/jaro_winkler functions the error "no module called Jellyfish found" is thrown

These functions are dependent on the Jellyfish package and this may not be installed on the executors used in your spark session.
Try submitting Jellyfish to your sparkcontext via addPyFile() or by setting the following environmental variables in your CDSW engine settings (ONS only):

* PYSPARK_DRIVER_PYTHON = /usr/local/bin/python3.6
* PYSPARK_PYTHON = /opt/ons/virtualenv/miscMods_v4.04/bin/python3.6

### Using the cluster function

The cluster function uses Graphframes, which requires an extra JAR file dependency to be submitted to your spark context in order for it to run.

We have published a graphframes-wrapper package on Pypi that contains this JAR file. This is included in the package requirements
as a dependency.

If outside of ONS and this dependency doesn't work, you will need to submit graphframes' JAR file dependency to your spark context. This can be found here:

https://repos.spark-packages.org/graphframes/graphframes/0.6.0-spark2.3-s_2.11/graphframes-0.6.0-spark2.3-s_2.11.jar

Once downloaded, this can be submitted to your spark context by adding this parameter to your SparkSession config: 

```sh
spark.conf.set('spark.jars', path_to_jar_file)
```

## Thanks

Thanks to all those in the Data Linkage Hub, Data Engineering and Methodology at ONS that have contributed towards this repository.
