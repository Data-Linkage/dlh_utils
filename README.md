# DLH_utils

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![PyPi Version](https://badge.fury.io/py/dlh-utils.svg)](https://pypi.org/project/dlh-utils/)
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

## Using the cluster function

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
