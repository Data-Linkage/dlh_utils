#temporary file to build and test date_diff, which will go back into dataframes.py
#when complete

import pyspark.sql.functions as F
#import to_timestamp, col, datediff, months_between
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

spark = (
    SparkSession.builder.appName("dataframe_testing")
    .config("spark.executor.memory", "5g")
    .config("spark.yarn.excecutor.memoryOverhead", "2g")
    .getOrCreate()
)


##################function, to go to dataframes.py

def date_diff(df, col_name1, col_name2, diff='Difference',
              in_date_format='dd-MM-yyyy', units='days', absolute=True):

  """
  IMPORTANT. Please note that Spark uses the Java string format for dates. This means that
  30/09/2001 should be written as in_date_format= 'dd/MM/yyyy'. Using lowercase m for months will be interpreted
  by Spark as minutes. 
  See: https://docs.oracle.com/javase/10/docs/api/java/time/format/DateTimeFormatter.html
  And: https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

  Rest of docstring, to import from existing function and add tests

  """

  #df = test_df
  #diff='Difference'
  #in_date_format='dd/MM/yyyy'
  #col_name1 = 'date1'
  #col_name2 = 'date2'
  #units = 'months'
  #absolute=True

  #convert string dates to Spark timestamps
  df = df.withColumn(f'{col_name1}_timestamp', F.to_timestamp(F.col(col_name1),in_date_format))
  df = df.withColumn(f'{col_name2}_timestamp', F.to_timestamp(F.col(col_name2),in_date_format))

  if units == 'days':
    df = df.withColumn(diff, F.round(F.datediff(F.col(f'{col_name2}_timestamp'),F.col(f'{col_name1}_timestamp')),2))
  elif units == 'months':
    df = df.withColumn(diff, F.round(F.months_between(F.col(f'{col_name2}_timestamp'),F.col(f'{col_name1}_timestamp')),2))
  elif units == 'years':
    df = df.withColumn(diff, F.round(F.months_between(F.col(f'{col_name2}_timestamp'),F.col(f'{col_name1}_timestamp'))/F.lit(12),2))
  else:
    raise Exception("units must be 'days', 'months' or 'years'") 

  if absolute:
    df = df.withColumn(diff, F.abs((F.col(diff))))
    
  #drop the timestamps
  df = df.drop(f'{col_name1}_timestamp',f'{col_name2}_timestamp')
    
  return df








##################tests, to go to test_dataframes.py

################## test 1 here

test_schema = StructType(
    [
        StructField("date1", StringType(), True),
        StructField("date2", StringType(), True),
    ]
    )

test_data = [
    ['01/02/2000','01/02/2023'],
    ['20/02/2010','05/08/2023'],
    [None,'10/10/2023'],
    ['02/03/1066','01/11/2023'],
    ['28/12/1987','15/12/2023'],
    ]

test_df = spark.createDataFrame(test_data, test_schema)

#run function on test
date_diff(test_df,'date1','date2',in_date_format='dd/MM/yyyy',units='days').show()
#############################

###put additional tests here, change names to test_schema2 etc