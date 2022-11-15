# DLH utils

A package produced by the linkage development team from the Data Linkage Hub, containing a set of functions used to expedite and streamline the data linkage process.

Thanks to all those in the Data Linkage Hub, Data Engineering and Methodology that have contributed towards this repository.

Please log an issue on the issue board or contact any of the active contributors with any issues or suggestions for improvements you have.

## Installation steps

* click the 'clone' button on the project homepage and copy the project's HTTP address
* open a terminal session within CDSW and run `git clone [http_address]`
* the project files will now be moved to your local file structure, within a folder called "dlh_utils"
* you can now install the package, typically by running either `!pip3 install '/home/cdsw/dlh_utils'` in a workbench/jupyter notebook session, or `pip3 install '/home/cdsw/dlh_utils'` in terminal. 

**Note: the filepath shown in this example may differ depending on where you have cloned the project.**
* all finished! You can now import modules from the dlh_utils package like any other Python library

*This package is a work in progress!* We will notify you of significant changes to the package. If you want to upgrade to the latest version, clone the project from GitLab again and run either `!pip3 install -U '[path_to_dlh_utils]'` in workbench, or `pip3 install -U '[path_to_dlh_utils]'` in terminal, to upgrade your package installation.

## Using the cluster function

The cluster function uses Graphframes, which requires an extra JAR file dependency to be submitted to your spark context in order for it to run.

We have published a graphframes-wrapper package on Pypi that contains this JAR file. This is included in the package requirements
as a dependency.

If outside of ONS and this dependency doesn't work, you will need to submit graphframes' JAR file dependency to your spark context. This can be found here:

https://repos.spark-packages.org/graphframes/graphframes/0.6.0-spark2.3-s_2.11/graphframes-0.6.0-spark2.3-s_2.11.jar

Once downloaded, this can be submitted to your spark context via: `spark.conf.set('spark.jars', path_to_jar_file)`
