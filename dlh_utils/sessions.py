'''
Function used to create and start different sized spark sessions, also
generating a Spark UI link to monitor session progress.
'''
import os
from IPython.core.display import display, HTML
from pyspark.sql import SparkSession
import graphframes_jars as graphframes


def getOrCreateSparkSession(appName='DE_DL',
                            size='large',
                            showConsoleProgress='false',
                            shufflePartitions=200,
                            defaultParallelism=200,
                            memory='10g',
                            memoryOverhead='1g',
                            cores=5,
                            maxExecutors=5):
    """
    Starts spark session dependent on size category specified
    or starts custom session on specified parameters. Also generates
    Spark UI link in console for monitoring session progress/resource use.

    Parameters
    ----------
    appName : str
      The name of the spark session
    size: {'small','medium','large','extra_large','custom'},default = 'large'
      The size category of session to be started
    showConsoleProgress ; {True, False}, default = 'false'
      Option to display UI metrics in console
    shufflePartitions : int, default = 200
      The default number of partitions to be used in repartitioning
    defaultParallelism: int, default = 200
      Default number of partitions in resilient distributed datasets (RDDs)
      returned by transformations like join, reduceByKey, and parallelize
      when no shufflePartition number is set.
    memory : str, default = 10g(GB)
      Executor memory allocation
    memoryOverhead : str, default = 1g(GB)
      The amount of off-heap memory to be allocated per driver in cluster mode
    cores : int, default = 5
      The number of cores to use on each executor
    maxExecutors : int, default = 5
      Upper bound for the number of executors

    Returns
    -------
     sparkSession and Spark UI web link in workbench console

    Raises
    -------
      None at present.
    """
    # obtain spark UI url parameters
    url = 'spark-' + str(os.environ['CDSW_ENGINE_ID']) + \
        '.'+str(os.environ['CDSW_DOMAIN'])
    spark_ui = display(HTML(f'<a href=http://{url}s>Spark UI</a>'))

    try:

      # get graphframes jar path to configure session with
        graphframes_path = graphframes.__file__
        graphframes_path = graphframes_path.rsplit('/', 1)[0]

        for file in os.listdir(graphframes_path):
            if file.endswith(".jar"):
                # Get the latest jar file
                jar_path = os.path.join(graphframes_path, file)

    except FileNotFoundError:
        print("graphframes wrapper package not found.\
              Please install this to use the cluster_number() function.")
        jar_path = None

    if size == 'small':

        spark = (
            SparkSession.builder.appName(appName)
            .config("spark.executor.memory", "1g")
            .config("spark.executor.cores", 1)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 3)
            .config("spark.sql.shuffle.partitions", 12)
            .config("spark.jars", jar_path)
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.ui.showConsoleProgress", showConsoleProgress)
            .enableHiveSupport()
            .getOrCreate()
        )

    if size == 'medium':

        spark = (
            SparkSession.builder.appName(appName)
            .config("spark.executor.memory", "6g")
            .config("spark.executor.cores", 3)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 3)
            .config("spark.sql.shuffle.partitions", 18)
            .config("spark.jars", jar_path)
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.ui.showConsoleProgress", showConsoleProgress)
            .enableHiveSupport()
            .getOrCreate()
        )

    if size == 'large':

        spark = (
            SparkSession.builder.appName(appName)
            .config("spark.executor.memory", "10g")
            .config("spark.yarn.executor.memoryOverhead", "1g")
            .config("spark.executor.cores", 5)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 5)
            .config("spark.jars", jar_path)
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.ui.showConsoleProgress", showConsoleProgress)
            .enableHiveSupport()
            .getOrCreate()
        )

    if size == 'extra_large':

        spark = (
            SparkSession.builder.appName(appName)
            .config("spark.executor.memory", "20g")
            .config("spark.yarn.executor.memoryOverhead", "2g")
            .config("spark.executor.cores", 5)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 12)
            .config("spark.jars", jar_path)
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.ui.showConsoleProgress", showConsoleProgress)
            .enableHiveSupport()
            .getOrCreate()
        )

    if size == 'custom':

        spark = (
            SparkSession.builder.appName(appName)
            .config('spark.executor.memory', memory)
            .config('spark.executor.memoryOverhead', memoryOverhead)
            .config('spark.executor.cores', cores)
            .config('spark.dynamicAllocation.maxExecutors', maxExecutors)
            .config("spark.sql.shuffle.partitions", shufflePartitions)
            .config("spark.default.parallelism", defaultParallelism)
            .config('spark.ui.showConsoleProgress', showConsoleProgress)
            .config("spark.jars", jar_path)
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.dynamicAllocation.enabled", "true")
            .getOrCreate()
        )

    return spark
