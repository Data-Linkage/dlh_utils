import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
import pyspark.sql.functions
from pyspark.sql.functions import col, sha2, lit, concat, substring, expr, length
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

import numpy as np

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



###############################################################################



def hash_id(df, column, salt):
  """
  Deletes your id column from your dataframe and replaces it with a 
  salted-hashed version of that id. 
  
  It also creates a lookup table of the original id column and it's 
  associated hashed variable.
  
  Parameters
  ----------
  df: dataframe
    Dataframe to which the function is applied.
  column: id column (str)
    The name of the id column you wish to be hashed.
    Input column as a string.
  salt: word of your choice
    The salt technique concatinates the desired word to the start of 
    each id field. The salt word is to be stored in the DAP working area 
    only and not shared with anybody who doesn't have access to the DAP 
    working area. The salt technique is designed to make the hashed id 
    harder to unhash.
    Input salt as a string.
    
  Returns
  -------
  The function returns a tuple of two dataframes:
    lookup_table
      The function returns a lookup table of the id value and its 
      associated hashed value to keep for reference.
    df
      The function also returns the original dataframe with the id 
      column replaced by the associated hashed column.

  Example
  -------

  > df.show()
  +---+--------+------+--------+
  | id|     dob|  name|postcode|
  +---+--------+------+--------+
  | a1|19990121|SPORTY| PO356TH|
  | a2|19771109| SCARY|   E34TO|
  | a3|19760424|  BABY| BH242ED|
  | a4|19251217|GINGER| KN231SD|
  | a5|19730724|  POSH| HK192YC|
  +---+--------+------+--------+
    
  As the function returns a tuple, call in the function using tuple unpacking
    
  > id_lookup,hashed_df = hash_id(df,column='id',salt='ninetiesspices') 
  
  Output
    the name of your salt is: ninetiesspices
  
  > id_lookup.show(truncate=False)
  +---+----------------------------------------------------------------+
  |id |hashed_id                                                       |
  +---+----------------------------------------------------------------+
  |a1 |4211b617e706b6e2d0dcee6f9d9cd4ac244fd2c8ccc8782f15239b15e4301898|
  |a2 |58f6c6926db272b9e4b6c12a70ba1b955540f2fea36336ad84147c74593fa448|
  |a3 |8c96ed6856d3c1056c7108d721e21292e776e61f93de7bc224f767623f72a5e3|
  |a4 |73dbff8492726e680f28b2dfd429f018fbf8c8e1b7230cf271cc81846a059fdd|
  |a5 |a9bd8fba7b9cbf2eeb20f22cae24cc6e1378294c07dd777fa7db48f9b97f9785|
  +---+----------------------------------------------------------------+
  
  > hashed_df.show(truncate=False)
  +--------+------+--------+----------------------------------------------------------------+
  |dob     |name  |postcode|hashed_id                                                       |
  +--------+------+--------+----------------------------------------------------------------+
  |19990121|SPORTY|PO356TH |4211b617e706b6e2d0dcee6f9d9cd4ac244fd2c8ccc8782f15239b15e4301898|
  |19771109|SCARY |E34TO   |58f6c6926db272b9e4b6c12a70ba1b955540f2fea36336ad84147c74593fa448|
  |19760424|BABY  |BH242ED |8c96ed6856d3c1056c7108d721e21292e776e61f93de7bc224f767623f72a5e3|
  |19251217|GINGER|KN231SD |73dbff8492726e680f28b2dfd429f018fbf8c8e1b7230cf271cc81846a059fdd|
  |19730724|POSH  |HK192YC |a9bd8fba7b9cbf2eeb20f22cae24cc6e1378294c07dd777fa7db48f9b97f9785|
  +--------+------+--------+----------------------------------------------------------------+
    """
  
  df = df.withColumn(
  'hashed_'+column,
    sha2(
      concat(
        lit(salt),
        col(column)
      ),
      256)
  )

  lookup_table = df.select(
    col(column),
    col('hashed_'+column)
  ).na.drop()

  df = df.drop(column)

  print(f'the name of your salt is: {salt}')
  
  return lookup_table, df

###############################################################################


def date_year_quarter(df, date_column):
  """
  Deletes your date column and replaces it with 
  a year column and a year quarter column.
  
  The column can be string, integer or datestamp.
  
  The function will output 'input_error' if 
  the length of numbers in the date_column is not equal to 8
  or the month in the date is not 01-12.
  
  prerequisites
  -------------
  Please insure that the date_column is fully numeric and the 
  order is year then month then day e.g yyyyMMdd, yyyy-MM-dd etc...
  Please insure that the month is represented as two numbers
  i.e Janurary would be represented at 01 instead of 1
  
    Parameters
  ----------
  df: dataframe
    Dataframe to which the function is applied.
  date_column: date column (str)
    The name of the date column that you want to change to year and year quarter.
    
  Returns
  -------
  The function returns your dataframe but with the date column omitted and  
  replaced with a date year and a date year quarter column.
  
  Example
  -------

  > df.show()
  +---+-----------+------+--------+
  | id|        dob|  name|postcode|
  +---+-----------+------+--------+
  | a1|QQ 123456 C|SPORTY| PO356TH|
  | a2|07451224152| SCARY|KZ66 ZYT|
  | a3|   19760424|  BABY| BH242ED|
  | a4|   19750127|GINGER| KN231SD|
  | a5|   19730724|  POSH| HK192YC|
  +---+-----------+------+--------+
  
  > date_year_quarter(df, date_column='dob').show()
  
  +---+------+--------+-----------+----------------+
  | id|  name|postcode|   dob_year|dob_year_quarter|
  +---+------+--------+-----------+----------------+
  | a1|SPORTY| PO356TH|input_error|     input_error|
  | a2| SCARY|KZ66 ZYT|input_error|     input_error|
  | a3|  BABY| BH242ED|       1976|              Q2|
  | a4|GINGER| KN231SD|       1975|              Q1|
  | a5|  POSH| HK192YC|       1973|              Q3|
  +---+------+--------+-----------+----------------+

  """
  
  df = df.withColumn(
    date_column,
    col(date_column).cast(
      StringType()
    )
  )
  
  df = df.withColumn(
    date_column,
    regexp_extract(
      date_column,
      "[0-9]+",
      idx=0)
  )
  
  df = df.withColumn(
    f'{date_column}_year',
    col(date_column).substr(
      1,4)
  )
  
  df = df.withColumn(
    f'{date_column}_year_quarter',
    col(date_column).substr(
      5,2)
  )
  
  replace_dict = {'01': 'Q1',
                  '02': 'Q1',
                  '03': 'Q1',
                  '04': 'Q2',
                  '05': 'Q2',
                  '06': 'Q2',
                  '07': 'Q3',
                  '08': 'Q3',
                  '09': 'Q3',
                  '10': 'Q4',
                  '11': 'Q4',
                  '12': 'Q4',}
  
      
  df = df.na.replace(replace_dict, 
                     subset=f'{date_column}_year_quarter')
 
  for i in [f'{date_column}_year', f'{date_column}_year_quarter']:
    df = df.withColumn(i, 
                       when(~col(f'{date_column}_year_quarter').isin(['Q1','Q2','Q3','Q4']), 
                             'input_error')
                       .when(length(col(date_column)) != 8, 
                             'input_error')
                           .otherwise(col(i)))
  
  df = df.drop(date_column)
  
  return df


###############################################################################
