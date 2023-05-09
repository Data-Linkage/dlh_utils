#The three ways to create a spark dataframe, as used in dlh_utils tests

#In most cases, a spark df is created when data are read in, but in tests
#we need to create small test dataframes to explore the function behaviour


#Before that, first open a spark session, to enable creating a spark df
from pyspark.sql import SparkSession
spark = (
    SparkSession.builder.appName("dataframe_testing")
    .config("spark.executor.memory", "5g")
    .config("spark.yarn.excecutor.memoryOverhead", "2g")
    .getOrCreate()
)




#1 Convert from a Pandas df:

#imports normally go at the top of a file, but i'm doing next to each piece of
#code, so you can see why we need it
import pandas as pd

df_1 = spark.createDataFrame(
    (
        pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "middlename": ["Maria", None, "", "Greg"],
                "lastname": ["Jones", None, "Smith", "Evans"],
                "numeric": [1, 2, None, 4],
                "after": [
                    "Maria_Jones",
                    "Claire",
                    "Josh_Smith",
                    "Bob_Greg_Evans",
                ],
            }
        )
    )
)



#2 Create with explicit schema, directly in Pyspark:
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

intended_schema = StructType(
    [
        StructField("firstname", StringType(), True),
        StructField("middlename", StringType(), True),
        StructField("lastname", StringType(), True),
        StructField("numeric", DoubleType(), True),
        StructField("after", StringType(), True),
        StructField("fullname", StringType(), True),
    ]
)

intended_data = [
    [None, "Maria", "Jones", 1.0, "Maria_Jones", "Maria_Jones"],
    ["Claire", None, None, 2.0, "Claire", "Claire"],
    ["Josh", "", "Smith", None, "Josh_Smith", "Josh_Smith"],
    ["Bob", "Greg", "Evans", 4.0, "Bob_Greg_Evans", "Bob_Greg_Evans"],
]

df_2 = spark.createDataFrame(intended_data, intended_schema)


#3 Create with implied schema (don't do this in real life)

df_3 = spark.createDataFrame(
    [
        (None, 3, 6),  
        (None, 1, 2),
        ("text", 4, 8),
        (None, None, 2),
    ],
    ["col1", "col2", "col3"]  # add your column names here
)


#Have a look at them: (change 1 to 2, 3)
df_1.show()

df_1.printSchema()