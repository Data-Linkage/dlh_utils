import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
import pyspark.sql.functions
from pyspark.sql.functions import regexp_extract, col, sha2, lit 
from pyspark.sql.functions import concat, substring, expr, length
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

import pandas as pd
import numpy as np
import os
import random

spark = (
    SparkSession.builder.appName("small-session")
    .config("spark.executor.memory", "1g")
    .config("spark.executor.cores", 1)
    .config("spark.dynamicAllocation.enabled", "true")
    .config("spark.dynamicAllocation.maxExecutors", 3)
    .config("spark.sql.shuffle.partitions", 12)
    .config("spark.shuffle.service.enabled", "true")
    .config("spark.ui.showConsoleProgress", "false")
    .config("spark.sql.repl.eagerEval.enabled", "true")
    .enableHiveSupport()
    .getOrCreate()
)

#########################---- datasets START -----##########################

data = [("a1", "QQ 123456 C", "SPORTY", "PO356TH"),
("a2", "19830914", "SCARY", "KZ66 ZYT"),
("a3", "19760424", "BABY", "B242ED"),
("a4", "19750127", "GINGER", "KN231SD"),
("a5", "19730724", "POSH", "HK192YC"),
("ABE46372837", "19631011", "SKINNER", "SP63HW"),
("a7", "19780306", "FLANDERS", "6878844321"),
("a8", "07451224152", "BARNEY", "PO29 2YC"),
("a9", "19630507", "KRUSTY", "NP65LN"),
("a10", "19981218", "MILHOUSE", "MK17 6PO")
       ]

schema = StructType([
    StructField("id", StringType(), True),
    StructField("dob", StringType(), True),
    StructField("name", StringType(), True),
    StructField("postcode", StringType(), True)
])

df = spark.createDataFrame(data = data, schema = schema)

#########################---- datasets END -----##########################


def identifying_strings(df, required_identifiers):
  """
  You can choose which regex expressions you want to search for in the 
  regex database from the data folder.
  
  Returns your dataframes with a corresponding identifying column for 
  each column in the dataframe. In the corresponding identifying column, 
  It will say if a regex expression was found and which regex expression 
  was used to find it.
  
  It also prodives a database of the regex expressions you have chosen 
  for you to document as the regex expressions may change over time.
  
  Parameters
  ----------
  df: dataframe
    Dataframe to which the function is applied.
  required_identifiers: regex expressions you want to search for
    unput the regex expressions that you want to search for as a 
    list of strings
  
    
  Returns
  -------
  The function returns a tuple of two dataframes:
    found identifying strings
      Returns your dataframes with a corresponding identifying column for 
      each column in the dataframe. In the corresponding identifying column, 
      It will say if a regex expression was found and which regex expression 
      was used to find it.
    regex database
      A database of the regex expressions you have used in this function 
      for you to document

  Example
  -------
  
  > df.show()
  +-----------+-----------+--------+----------+
  |         id|        dob|    name|  postcode|
  +-----------+-----------+--------+----------+
  |         a1|QQ 123456 C|  SPORTY|   PO356TH|
  |         a2|   19830914|   SCARY|  KZ66 ZYT|
  |         a3|   19760424|    BABY|    B242ED|
  |         a4|   19750127|  GINGER|   KN231SD|
  |         a5|   19730724|    POSH|   HK192YC|
  |ABE46372837|   19631011| SKINNER|    SP63HW|
  |         a7|   19780306|FLANDERS|6878844321|
  |         a8|07451224152|  BARNEY|  PO29 2YC|
  |         a9|   19630507|  KRUSTY|    NP65LN|
  |        a10|   19981218|MILHOUSE|  MK17 6PO|
  +-----------+-----------+--------+----------+

  

  > search_for = ['NIN', 
              'UK Vehicle Reg',
              'UK phone no',
              'NHS No',
              'UK Child Benefit']

  > identified, regex_used = identifying_strings(df=df, 
                    required_identifiers=search_for)
                    
  > identified.show()
  +-----------+-----------+--------+----------+----------------+--------------+---------------+-------------------+
  |         id|        dob|    name|  postcode|   id_identifier|dob_identifier|name_identifier|postcode_identifier|
  +-----------+-----------+--------+----------+----------------+--------------+---------------+-------------------+
  |         a8|07451224152|  BARNEY|  PO29 2YC|            null|   UK phone no|           null|               null|
  |         a7|   19780306|FLANDERS|6878844321|            null|          null|           null|             NHS No|
  |         a2|   19830914|   SCARY|  KZ66 ZYT|            null|          null|           null|     UK Vehicle Reg|
  |         a1|QQ 123456 C|  SPORTY|   PO356TH|            null|           NIN|           null|               null|
  |ABE46372837|   19631011| SKINNER|    SP63HW|UK Child Benefit|          null|           null|               null|
  +-----------+-----------+--------+----------+----------------+--------------+---------------+-------------------+
  
  > spark.createDataFrame(regex_used).show()
  +----------------+--------------------+--------------+-----+
  |      Identifier|               Regex|        Origin|Notes|
  +----------------+--------------------+--------------+-----+
  |          NHS No|            ^\d{10}$|      de_utils|  NaN|
  |             NIN|^\s*[a-zA-Z]{2}(?...|Stack Overflow|  NaN|
  |     UK phone no|^(?:0|\+?44)(?:\d...|  RegExLib.com|  NaN|
  |UK Child Benefit|     ^[A-Z]{3}\d{8}$|  RegExLib.com|  NaN|
  |  UK Vehicle Reg|^([A-Z]{3}\s?(\d{...|  RegExLib.com|  NaN|
  +----------------+--------------------+--------------+-----+
  
  """
  
  identifier_dataset = pd.read_csv('Data/regex_database.csv')
  
  regex_dict = dict(zip(identifier_dataset.Identifier,
                      identifier_dataset.Regex))

  required_dict = {key: regex_dict[key] for key in regex_dict.keys()
                        & {i for i in required_identifiers}}

  required_regex = list(required_dict.values())

  new_df = spark.createDataFrame([], schema=df.schema)
    
  for i in df.columns:
    for l in required_regex:
      filtered = df.filter(col(i).rlike(l))
      new_df = new_df.union(filtered)

  for n in new_df.columns:
    new_df = new_df.withColumn(n+'_identifier', lit(None))
    for k in required_dict:
      new_df = new_df.withColumn(n+'_identifier',
                                 when(col(n).rlike(required_dict.get(k)), k)
                                 .otherwise(col(n+'_identifier')))     
  
  new_df = new_df.distinct()
  
  pd.set_option("display.max_rows", None)
  pd.set_option('display.max_colwidth', None)
  regex_used = identifier_dataset[identifier_dataset['Identifier']
                                  .isin(required_identifiers)]
  
  return new_df, regex_used


search_for = ['NIN', 
              'UK Vehicle Reg',
              'UK phone no',
              'NHS No',
              'UK Child Benefit']

identified, regex_used = identifying_strings(df=df, 
                    required_identifiers=search_for)
