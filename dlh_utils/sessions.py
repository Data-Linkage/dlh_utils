import os
from IPython.core.display import display, HTML
from pyspark.sql import SparkSession
graphframes = __import__('graphframes-wrapper')

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
    spark_ui = display(HTML('<a href=http://%s>Spark UI</a>' % url))

    try:
      
      # get graphframes jar path to configure session with
      graphframes_path = graphframes.__file__
      graphframes_path = graphframes_path.rsplit('/', 1)[0]

      for file in os.listdir(graphframes_path):
          if file.endswith(".jar"):
              # Get the latest jar file
              jar_path = os.path.join(graphframes_path, file)
              
    except FileNotFoundError:
      print("graphframes wrapper package not found. Please install this to use the cluster_number() function.")
      jar_path = None
      
    if size == 'small':

        """
        Small Session

        This session is similar to that used for DAPCATS training
        It is the smallest session that is realistically used

        Details:
            Only 1g of memory and 3 executors
            Only 1 core
            Number of partitions are limited to 12, which can improve
            performance with smaller data

        Use case:
            Simple data exploration of small survey data

        Example of actual usage:
            Used for DAPCATS PySpark training
            Mostly simple calculations
        """

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

        """
        Medium Session

        A standard session used for analysing survey or synthetic
        datasets. Also used for some Production pipelines based on
        survey and/or smaller administrative data.

        Details:
            6g of memory and 3 executors
            3 cores
            Number of partitions are limited to 18, which can improve
            performance with smaller data

        Use case:
            Developing code in Dev Test
            Data exploration in Production
            Developing Production pipelines on a sample of data
            Running smaller Production pipelines on mostly survey data

        Example of actual usage:
            Complex calculations, but on smaller synthetic data in
                Dev Test
        """

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

        """
        Large Session

        Session designed for running Production pipelines on large
        administrative data, rather than just survey data. Will often
        develop using a smaller session then change to this once the
        pipeline is complete.

        Details:
            10g of memory and 5 executors
            1g of memory overhead
            5 cores, which is generally optimal on larger sessions

        Use case:
            Production pipelines on administrative data
            Cannot be used in Dev Test, as 9 GB limit per executor

        Example of actual usage:
            One administrative dataset of 100 million rows
            Many calculations
        """

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

        """
        Extra Large session

        Used for the most complex pipelines, with huge administrative
        data sources and complex calculations. Uses a large amount of
        resource on the cluster, so only use when running Production
        pipelines

        Details:
            20g of memory and 12 executors
            2g of memory overhead
            5 cores; using too many cores can actually cause worse
                performance on larger sessions

        Use case:
            Running large, complex pipelines in Production on mostly
                administrative data
            Do not use for development purposes; use a smaller session
                and work on a sample of data or synthetic data

        Example of actual usage:
            Three administrative datasets of around 300 million rows
            Significant calculations, including joins and writing/reading
                to many intermediate tables
        """

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

        """
        Optional custom spark settings
        """

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
