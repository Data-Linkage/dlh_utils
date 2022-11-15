'''
Utility functions used to ease difficulty in querying databases and produce descriptive
metrics about a dataframe
'''
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.context import SparkContext as sc
import pandas as pd
import os
import errno
import re
from dlh_utils import dataframes as da
from dlh_utils import utilities as ut

###############################################################################


def list_files(directory, walk=False):
    """
    Lists files in a given HDFS directory, and/or all of that directory's
    subfolders if specified


    Parameters
    ---------- 
    directory : str
      String path of directory
    walk : boolean {True, False}
      Lists files only in immediate directory specified if walk = False. 
      Lists all files in immediate directory and all subfolders if Walk = True

    Returns
    -------
    list
      List of files

    Raises
    -------
      None at present.

    Example
    -------
    > list_files(directory = '/dev/kicktyres/',walk=False)

    ['hdfs://prod1/dev/kicktyres/hive_tables',
     'hdfs://prod1/dev/kicktyres/shakes',
     'hdfs://prod1/dev/kicktyres/shakes_count',
     'hdfs://prod1/dev/kicktyres/timings_tables']

    > list_files(directory = '/dev/kicktyres/',walk=True)

    ['hdfs://prod1/dev/kicktyres/shakes_count/part-r-00111',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00164',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00190',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00039',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00202',
     'hdfs://prod1/dev/kicktyres/shakes',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00036',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00183',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00193',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00128',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00005',
     'hdfs://prod1/dev/kicktyres/shakes_count/part-r-00057',
     ... and all other files contained within these sub-directories

    """

    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration

    host = 'user'

    fs = FileSystem.get(URI(host), Configuration())

    status = fs.listStatus(Path(directory))

    files = [str(fileStatus.getPath()) for fileStatus in status]

    if walk == False:

        return files

    else:

        for file in files:
            if len(files) == len(set(files)):
                files.extend(list_files(file))
            else:
                break

        files = list(set(files))

        return files

###############################################################################


def list_checkpoints(checkpoint):
    """
    Lists checkpoints in HDFS directory

    Parameters
    ---------- 
    checkpoint : str
      String path of checkpoint directory

    Returns
    -------
    list
      List of files in checkpoint directory

    Raises
    -------
      None at present.

    Example
    -------

    > list_checkpoints(checkpoint = '/user/edwara5/checkpoints')

    ['hdfs://prod1/user/edwara5/checkpoints/0299d46e-96ad-4d3a-9908-c99b9c6a7509/connected-components-985ca288']
    """

    return list_files(
        list_files(checkpoint, walk=False)[0])

###############################################################################


def list_tables(database):
    """
    Returns the tables in a database from hive, it takes an argument of the 
    database name as a string. It then returns a dataframe listing the tables
    within the database given.

    Parameters
    ---------- 
    database : str
      String name of database

    Returns
    -------
    list
      List of tables in database

    Raises
    -------
      None at present.

    Example
    -------

    > list_tables('baby_names')

    ['baby_names_boy_raw',
     'baby_names_boy_std',
     'baby_names_girl_raw',
     'baby_names_girl_std',
     'bv_girl_names_raw',
     'bv_girl_names_std']
    """

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql(f"SHOW TABLES IN {database}")
    return list((df
                 .select("tableName")
                 .toPandas()
                 )["tableName"])

###############################################################################


def most_recent(path, filetype, regex=None):
    """
    Returns most recently edited Hive table or directory containing most recently edited
    csv/parquet file(s) in location or database.

    Parameters
    ---------- 
    path : str
      The path or database which will be searched
    filetype : {csv, parquet, hive}
      The format of data that is to be searched for
    regex : str, optional
      A regular expression to filter the search results by, e.g. '^VOA'

    Returns
    -------
    most_recent_filepath: str
      Filepath or table reference for most recent data
    filetype: str
      The format of the data for which a filepath has been returned

    Raises
    -------
      FileNotFoundError if search query does not exist in HDFS. 

    Example
    -------

    > most_recent(path = 'baby_names', filetype = 'hive', regex = None)

    ('baby_names.bv_girl_names_raw', 'hive')

    > most_recent(path = 'baby_names', filetype = 'hive', regex = "std$")

    ('baby_names.bv_girl_names_std', 'hive')
    """

    # pass spark context to function
    spark = SparkSession.builder.getOrCreate()

    if regex == None:

        if filetype == 'hive':

            try:

                # list all tables in directory
                tables = spark.sql(
                    f"SHOW TABLES IN {path}").select('tableName')

                # create full filepath from directory & table name
                filepaths = tables.withColumn('path', F.concat(
                    F.lit(path), F.lit("."), F.col("tableName")))

                # convert to list
                filepaths = list(filepaths.select('path').toPandas()['path'])

                # initialise empty dictionary
                filepath_dict = dict()

                # loop through paths, appending path and time to dictionary
                for path in filepaths:

                    time = spark.sql(
                        f"SHOW tblproperties {path} ('transient_lastDdlTime')").collect()[0][0]

                    filepath_dict.update({path: time})

                # sort by max time since epoch and return corresponding path
                most_recent_filepath = max(
                    filepath_dict, key=filepath_dict.get)

            except:

                raise FileNotFoundError(
                    filetype + " file not found in this directory: " + path)

        # if filetype != hive
        else:

            # return all files in dir recursively, sorted by modification date (ascending), decode from bytes-like to str
            files = subprocess.check_output(
                ["hdfs", "dfs", "-ls", "-R", "-t", "-C", path]).decode()

            # split by newline to return list of old -> new files
            files = files.split('\n')

            if filetype == 'csv':

                try:

                    # filter for .csv ext and take last element of list
                    result = [f for f in files if f.endswith('csv')][-1]

                    # return path up until last '/'
                    most_recent_filepath = re.search('.*\/', result).group(0)

                except:

                    raise FileNotFoundError(
                        filetype + " file not found in this directory: " + path)

            elif filetype == 'parquet':

                try:

                    # filter for .csv ext and take last element of list
                    result = [f for f in files if f.endswith('parquet')][-1]

                    # return path up until last '/'
                    most_recent_filepath = re.search('.*\/', result).group(0)

                except:

                    raise FileNotFoundError(
                        filetype + " file not found in this directory: " + path)

    # if regex argument specified:
    else:

        if filetype == 'hive':

            try:

                # list all tables in directory
                tables = spark.sql(
                    f"SHOW TABLES IN {path}").select('tableName')

                # create full filepath from directory & table name
                filepaths = tables.withColumn('path', F.concat(
                    F.lit(path), F.lit("."), F.col("tableName")))

                # filter filepaths based on regex
                filtered_filepaths = filepaths.filter(
                    filepaths["path"].rlike(regex))

                # convert to list
                filtered_filepaths = list(
                    filtered_filepaths.select('path').toPandas()['path'])

                # initialise empty dictionary
                filepath_dict = dict()

                # loop through paths, appending path and time to dict
                for path in filtered_filepaths:

                    time = spark.sql(
                        f"SHOW tblproperties {path} ('transient_lastDdlTime')").collect()[0][0]

                    filepath_dict.update({path: time})

                # sort by max time since epoch and return corresponding path
                most_recent_filepath = max(
                    filepath_dict, key=filepath_dict.get)

            except:

                raise FileNotFoundError(filetype + " file, matching this regular expression: " + regex +
                                        " not found in this directory: " + path)

        # if filetype != hive
        else:

            # return all files in dir recursively, sorted by modification date (ascending), decode from bytes-like to str
            files = subprocess.check_output(
                ["hdfs", "dfs", "-ls", "-R", "-t", "-C", path]).decode()

            # split by newline to return list of old -> new files
            files = files.split('\n')

            r = re.compile(regex)

            # apply regex filter
            filtered_files = list(filter(r.match, files))

            if filetype == 'csv':

                try:

                    # filter for .csv ext and take last element of list
                    result = [
                        f for f in filtered_files if f.endswith('csv')][-1]

                    # return path up until last '/'
                    most_recent_filepath = re.search('.*\/', result).group(0)

                except:

                    raise FileNotFoundError(filetype + " file, matching this regular expression: " + regex +
                                            " not found in this directory: " + path)

            elif filetype == 'parquet':

                try:

                    # filter for .csv ext and take last element of list
                    result = [
                        f for f in filtered_files if f.endswith('parquet')][-1]

                    # return path up until last '/'
                    most_recent_filepath = re.search('.*\/', result).group(0)

                except:

                    raise FileNotFoundError(filetype + " file, matching this regular expression: " + regex +
                                            " not found in this directory: " + path)

    return most_recent_filepath, filetype

###############################################################################


def write_format(df, write, path,
                file_name=None, sep=",", header="true", mode='overwrite'):
    """
    Writes dataframe in specified format

    Can write data to HDFS in csv or parquet format and to database in hive table
    format.

    Parameters
    ---------- 
    df : dataframe
      Dataframe to be written
    write : {csv, parquet, hive}
      The format in which data is to be written
    path : str
      The path or database to which dataframe is to be written
    file_name : str
      The file or table name under which dataframe is to be saved. Note that if 
      None, function will write to the HDFS path specified in case of csv
      or parquet
    sep : str
      specified separator for data in csv format
    header : {True, False}
      Boolean indicating whether or not data will include a header
    mode : {overwrite, append}, default = overwrite
      Choice to overwrite existing file or table or to append new data into it

    Returns
    -------
    file or table
      Writen version of dataframe in specified format

    Raises
    -------
      None at present.

    Example
    -------

    > write_format(df = df, write = 'parquet', path = 'user/edwara5/simpsons.parquet',
                  mode = 'overwrite')
    """
    spark = SparkSession.builder.getOrCreate()
    if file_name == None:
        if write == 'csv':
            df.write.format('csv').option('header', header).mode(
                mode).option('sep', sep).save(f'{path}')
        if write == 'parquet':
            df.write.parquet(path=f'{path}', mode=mode)
    else:
        if write == 'csv':
            df.write.format('csv').option('header', header).mode(
                mode).option('sep', sep).save(f'{path}/{file_name}')
        if write == 'parquet':
            df.write.parquet(path=f'{path}/{file_name}', mode=mode)
        if write == 'hive':
            df.write.mode("overwrite").saveAsTable(f'{path}.{file_name}')

###############################################################################


def read_format(read, path=None, file_name=None,
               sep=",", header="true", inferSchema="True"):
    """
    Reads dataframe from specified format.

    Can read from HDFS in csv or parquet format and from database hive table 
    format.

    Parameters
    ---------- 
    read : str {csv, parquet, hive}
      The format from which data is to be read
    path : str (default = None)
      The path or database from which dataframe is to be read
    file_name : str (default = None)
      The file or table name from which dataframe is to be read. Note that if 
      None, function will read from HDFS path specified in case of csv
      or parquet
    sep : str
      specified separator for data in csv format
    header : {"true", "false"} (default = "true")
      Boolean indicating whether or not data will be read to include a header
    mode : {overwrite, append}, default = overwrite
      Choice to overwrite existing file or table or to append new data into it
    inferSchema : {"true", "false"}:
      Boolean indicating whether data should be read with infered data types and 
      schema. If false, all data will read as string format.

    Returns
    -------
    dataframe
      Dataframe of data read from specified path and format

    Raises
    -------
      None at present.

    Example
    -------

    > df = read_format(read = 'parquet', path = '/user/edwara5/simpsons.parquet', 
                      file_name = None, header= "true", inferSchema = "True")

    > df.show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ZZ99 9SZ|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ZZ99 5GB|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ZZ99 2SP|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ZZ99 2FA|
    +---+--------+----------+-------+----------+---+--------+
    """
    spark = SparkSession.builder.getOrCreate()
    if file_name == None:
        if read == 'csv':
            df = (spark.read.format('csv')
                  .option('sep', sep)
                  .option('header', header)
                  .option('inferSchema', inferSchema)
                  .load(f"{path}")
                  )
        if read == 'parquet':
            df = (spark.read.format('parquet')
                  .option('header', header)
                  .option('inferSchema', inferSchema)
                  .load(f"{path}")
                  )
    else:
        if read == 'csv':
            df = (spark.read.format('csv')
                  .option('sep', sep)
                  .option('header', header)
                  .option('inferSchema', inferSchema)
                  .load(f"{path}/{file_name}")
                  )
        if read == 'parquet':
            df = (spark.read.format('parquet')
                  .option('header', header)
                  .option('inferSchema', inferSchema)
                  .load(f"{path}/{file_name}")
                  )
        if read == 'hive':
            df = spark.sql(f"SELECT * FROM {path}.{file_name}")

    return df

###############################################################################


def search_files(path, string):
    """
    Finds file and line number(s) of specified string within a specified file
    path.

    Parameters
    ----------
    path : string
      Path directory for which the search function is applied to.
    string : string
      string value that is searched within the files of the directory given

    Returns
    -------
    dictionary
      Dictionary with keys of file names containing the string and values of
      line numbers indicating where there is a match on the string.

    Raises
    ------
    None at present.

    Example
    -------

    > search_files(path = '/home/cdsw/random_stuff', string = 'Homer')

    > {'simpsons.csv': [2]}

    """
    files_in_dir = os.listdir(path)
    diction = {}  # try empty dictionary

    for file in files_in_dir:
        count = 0
        countList = []

        try:
            with open(f'{path}/{file}') as f:
                datafile = f.readlines()

            for line in datafile:
                count = count + 1

                if string in line:
                    countList.append(count)

            if len(countList) != 0:
                diction[file] = countList
        except:
            continue

    return diction

###############################################################################


def describe_metrics(df, output_mode='pandas'):
    """
    Used to describe information about variables within a dataframe, including:
    * type
    * count
    * distinct value count
    * percentage of distinct values
    * null count
    * percentage of null values
    * non-null value count
    * percentage of non-null values

    Parameters
    ----------
    df : dataframe
      Dataframe to produce descriptive metrics about.
    output_mode: string, {'spark', 'pandas'}, default = pandas
      the type of dataframe to return

    Returns
    -------
    decribe_df
      A dataframe with columns detailing descriptive metrics on each variable

    Raises
    ------
    None at present.

    Example
    -------
    > describe_metrics(df = df,output_mode='spark').show()

    +----------+------+-----+--------+------------------+----+------------------+--------+-----------------+
    |  variable|  type|count|distinct|  percent_distinct|null|      percent_null|not_null| percent_not_null|
    +----------+------+-----+--------+------------------+----+------------------+--------+-----------------+
    |        ID|string|    6|       5| 83.33333333333334|   0|               0.0|       6|            100.0|
    |  Forename|string|    6|       5| 83.33333333333334|   0|               0.0|       6|            100.0|
    |Middlename|string|    6|       4| 66.66666666666666|   1|16.666666666666664|       5|83.33333333333334|
    |   Surname|string|    6|       1|16.666666666666664|   0|               0.0|       6|            100.0|
    |       DoB|string|    6|       5| 83.33333333333334|   0|               0.0|       6|            100.0|
    |       Sex|string|    6|       2| 33.33333333333333|   0|               0.0|       6|            100.0|
    |  Postcode|string|    6|       1|16.666666666666664|   0|               0.0|       6|            100.0|
    +----------+------+-----+--------+------------------+----+------------------+--------+-----------------+
    """

    spark = SparkSession.builder.getOrCreate()

    distinct_df = (df
                   .agg(*(F.countDistinct(F.col(c)).alias(c) for c in df.columns))
                   .withColumn('summary', F.lit('distinct')))
    null_df = (df
               .agg(*(F.count(F.when(F.isnan(F.col(c)) | F.col(c).isNull(), c))
                      .alias(c) for c in df.columns))
               .withColumn('summary', F.lit('null')))

    decribe_df = da.union_all(distinct_df, null_df).persist()

    count = df.count()

    types = df.dtypes
    types = dict(zip([x[0] for x in types],
                     [x[1] for x in types]))

    decribe_df = decribe_df.toPandas()
    decribe_df = decribe_df.transpose().reset_index()
    decribe_df.columns = ['variable']+list(decribe_df[decribe_df['index'] == 'summary']
                                           .reset_index(drop=True).transpose()[0])[1:]
    decribe_df = decribe_df[decribe_df['variable'] != 'summary']
    decribe_df['count'] = count
    decribe_df['not_null'] = decribe_df['count']-decribe_df['null']
    for variable in ['distinct', 'null', 'not_null']:
        decribe_df['percent_' +
                   variable] = (decribe_df[variable]/decribe_df['count'])*100
    decribe_df['type'] = [types[x] for x in decribe_df['variable']]

    decribe_df = decribe_df[[
        'variable',
        'type',
        'count',
        'distinct',
        'percent_distinct',
        'null',
        'percent_null',
        'not_null',
        'percent_not_null'
    ]]

    if output_mode == 'spark':
        decribe_df = ut.pandas_to_spark(decribe_df)

    return decribe_df

###############################################################################


def value_counts(df, limit=20, output_mode='pandas'):
    """
    Counts the most common values in all columns of a dataframe.

    Parameters
    ----------
    df : dataframe
      Dataframe to produce summary counts from.
    limit : integer, default = 20
      the top n values to search for.
    output_mode: string, {'spark', 'pandas'}, default = pandas
      the type of dataframe to return

    Returns
    -------
    None
      A dataframe with original dataframe columns and a count of 
      their most common values.

    Raises
    ------
    None at present.

    Example
    -------
    > value_counts(df = df, limit = 5, output_mode='spark').show()

    +---+--------+--------+--------------+----------+----------------+-------+-------------+
    | ID|ID_count|Forename|Forename_count|Middlename|Middlename_count|Surname|Surname_count|
    +---+--------+--------+--------------+----------+----------------+-------+-------------+
    |  3|       2|    Bart|             2|     Jo-Jo|               2|Simpson|            6|
    |  5|       1|   Homer|             1|      null|               1|       |            0|
    |  1|       1|   Marge|             1|    Juliet|               1|       |            0|
    |  4|       1|  Maggie|             1|     Marie|               1|       |            0|
    |  2|       1|    Lisa|             1|       Jay|               1|       |            0|
    +---+--------+--------+--------------+----------+----------------+-------+-------------+
    """
    spark = SparkSession.builder.getOrCreate()

    def value_count(df, col, limit):

        return (df.
                groupBy(col)
                .count()
                .sort('count', ascending=False)
                .limit(limit)
                .withColumnRenamed('count', col+'_count')
                .toPandas())

    dfs = [value_count(df, col, limit) for col in df.columns]

    def make_limit(df, limit):

        count = df.shape[0]

        if count < limit:

            dif = limit-count

            dif_df = pd.DataFrame({
                0: ['']*dif,
                1: [0]*dif
            })[[0, 1]]

            dif_df.columns = list(df)

            df = (df
                  .append(dif_df)
                  .reset_index(drop=True)
                  )

        return df

    dfs = [make_limit(df, limit) for df in dfs]

    df = pd.concat(dfs, axis=1)

    if output_mode == 'spark':

        df = ut.pandas_to_spark(df)

    return df

################################################################


def drop_hive_table(database, table_name):
    """
    Deletes hive table from Hive if it exists.

    Parameters
    ----------
    database : string
      Name of database.
    table_name : string
      Name of table.

    Returns
    -------
    None
      Drops Hive table.

    Raises
    ------
    None at present.
    """
    spark = SparkSession.builder.getOrCreate()

    spark.sql(f'DROP TABLE IF EXISTS {database}.{table_name}')

###################################################################


def clone_hive_table(database, table_name, new_table, suffix=''):
    """
    Duplicates hive table.

    Parameters
    ----------
    database : string
      Name of database.
    table_name : string
      Name of table being cloned.
    new_table :  string
      Name of cloned table.
    suffix : string, (default = '')
      string appended to table name.

    Returns
    -------
    None
      Clones table instead.

    Raises
    ------
    None at present.
    """
    spark = SparkSession.builder.getOrCreate()

    spark.sql(f'CREATE TABLE {database}.{new_table}{suffix} \
              AS SELECT * FROM {database}.{table_name}')

###################################################################


def rename_hive_table(database, table_name, new_name):
    """
    Renames Hive table.

    Parameters
    ----------
    database : string
      Name of database.
    table_name : string
      Name of table being renamed.
    new_name : string
      Name of new table.

    Returns
    -------
    None
      Renames Hive table.

    Raises
    ------
    None at present.
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sql(
        f'ALTER TABLE {database}.{table_name} RENAME TO {database}.{new_name}')

###################################################################


def create_hive_table(df, database, table_name):
    """
    Creates Hive table from dataframe.

    Saves all information within a dataframe into a Hive table.

    Parameters
    ----------
    df : dataframe
      Dataframe being saved as a Hive table.
    database : string
      Name of database Hive table is being saved to.
    table_name : string
      Name of table df is being named to.

    Returns
    -------
    None
      Saves data to Hive table.

    Raises
    ------
    None at present.
    """

    spark = SparkSession.builder.getOrCreate()

    df.createOrReplaceTempView("tempTable")
    spark.sql(f'CREATE TABLE {database}.{table_name} AS \
              SELECT * FROM tempTable')

###################################################################


def regex_match(df, regex, limit=10000, cut_off=0.75):
    """
    Returns a list of columns, for an input dataframe,
    that match a specified regex pattern.

    Parameters
    ----------
    df : dataframe
      Dataframe being searched for a text pattern.
    regex : string
      Regex pattern to match against
    limit : integer (default = 10000)
      Number of rows from dataframe to search for a
      text pattern
    cut_off : float (default = 0.75)
      The minimum rate of matching values in a column
      for it to be considered a regex match

    Returns
    -------
    list
      A list of all columns matching specified regex pattern.

    Raises
    ------
    None at present.

    Example
    -------
    > df.show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > regex_match(df = df,regex = "([A-Z])\w+",limit=5,cut_off=0.75)

    ['Forename', 'Middlename', 'Surname', 'Postcode']

    """

    sample_df = (df
                 .limit(limit)
                 ).persist()

    sample_df.count()

    counts_df = (sample_df
                 .groupBy()
                 .agg(*
                      [F.sum(F.when(F.col(col)
                                     .rlike(regex), 1)
                             ).alias(col)
                       for col in sample_df.columns]
                      )
                 )

    counts_df = (counts_df
                 .toPandas()
                 .transpose()
                 .dropna()
                 .reset_index()
                 .rename(columns={
                     'index': 'variable',
                     0: 'count',
                 })
                 )

    counts_df['match_rate'] = \
        counts_df['count']/limit

    counts_df = (counts_df
                 [counts_df['match_rate'] >= cut_off]
                 .reset_index(drop=True)
                 )

    sample_df.unpersist()

    return list(counts_df['variable'])

###################################################################


def pandas_to_spark(pandas_df):
    """
    Creates a spark dataframe from a given pandas dataframe

    Parameters
    ----------
    df : dataframe
      Pandas dataframe being converted.

    Returns
    -------
    df
      A spark dataframe

    Raises
    ------
    None at present.
    """
    def equivalent_type(f):

        if f == 'datetime64[ns]':
            return TimestampType()

        elif f == 'int64':
            return LongType()

        elif f == 'int32':
            return IntegerType()

        elif f == 'float64':
            return DoubleType()

        elif f == 'float32':
            return FloatType()

        else:
            return StringType()

    def define_structure(string, format_type):

        try:
            vartype = equivalent_type(format_type)

        except:
            vartype = StringType()

        return StructField(string, vartype)

    spark = SparkSession.builder.getOrCreate()

    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)

    struct_list = []

    for column, vartype in zip(columns, types):
        struct_list.append(define_structure(column, vartype))

    p_schema = StructType(struct_list)

    return spark.createDataFrame(pandas_df, p_schema)

###################################################################
