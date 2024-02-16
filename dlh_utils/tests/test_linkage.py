'''
Pytesting on Linkage functions.
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,IntegerType,\
DoubleType, FloatType
import pandas as pd
import pytest
from chispa import assert_df_equality
from dlh_utils.linkage import order_matchkeys,matchkey_join,extract_mk_variables,\
    assert_unique_matches,matchkey_counts,matchkey_dataframe,alpha_name, std_lev_score,\
    soundex,jaro,jaro_winkler, difflib_sequence_matcher, blocking, mk_dropna, \
    assert_unique, deterministic_linkage, clerical_sample, metaphone, cluster_number

pytestmark = pytest.mark.usefixtures("spark")

#############################################################################


class TestOrderMatchkeys(object):

    def test_expected(self,spark):

        dfo = spark.createDataFrame(
            (pd.DataFrame({
                "uprn": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                         '14', '15','16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        dffn = spark.createDataFrame(
            (pd.DataFrame({
                "uprn": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                         '14', '15','16', '17', '18', '19', '20'],

                "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        mks = [
            [F.substring(dfo['first_name'], 1, 1) == F.substring(dffn['first_name'], 1, 1),
             F.substring(dfo['last_name'], 1, 1) == F.substring(dffn['last_name'], 1, 1)],

            [F.substring(dfo['first_name'], 1, 1) == F.substring(dffn['first_name'], 1, 1),
                dfo['last_name'] == dffn['last_name']],
            [dfo['first_name'] == dffn['first_name'],
                dfo['last_name'] == dffn['last_name']]
        ]
        '''
        test_df = pd.DataFrame({
            'mks': mks,
            'count': [(dfo.join(dffn, on=mk, how='inner')).count()
                      for mk in mks]
        })
        '''
        intended_list = [
            [dfo['first_name'] == dffn['first_name'],dfo['last_name'] == dffn['last_name']],
            [F.substring(dfo['first_name'], 1, 1) == F.substring(dffn['first_name'], 1, 1),
                dfo['last_name'] == dffn['last_name']],
            [F.substring(dfo['first_name'], 1, 1) == F.substring(dffn['first_name'], 1, 1),
             F.substring(dfo['last_name'], 1, 1) == F.substring(dffn['last_name'], 1, 1)]
        ]

        result_list = order_matchkeys(dfo, dffn, mks)

        assert result_list[0] and intended_list[0]
        assert result_list[1] and intended_list[1]
        assert result_list[2] and intended_list[2]

##############################################################################


class TestMatchkeyJoin(object):

    def test_expected(self,spark):

        test_df_1 = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        test_df_2 = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        mks = [
            [test_df_1['first_name'] == test_df_2['first_name'],
             test_df_1['last_name'] == test_df_2['last_name']],

            [F.substring(test_df_1['first_name'], 1, 1) == F.substring(test_df_2['first_name'],
                                                                       1, 1),
                test_df_1['last_name'] == test_df_2['last_name']],

            [F.substring(test_df_1['first_name'], 1, 1) == F.substring(test_df_2['first_name'],
                                                                       1, 1),
                F.substring(test_df_1['last_name'], 1, 1) == F.substring(test_df_2['last_name'],
                                                                         1, 1)]
        ]

        intended_schema = StructType([
            StructField("l_id",StringType(),True),
            StructField("r_id",StringType(),True),
            StructField("matchkey",IntegerType(),False)
        ])

        intended_data = [['7', '7', 1],
                         ['15', '15', 1],
                         ['8', '8', 1],
                         ['5', '5', 1],
                         ['18', '18', 1],
                         ['9', '9', 1],
                         ['10', '10', 1],
                         ['12', '12', 1],
                         ['13', '13', 1],
                         ['14', '14', 1]]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = matchkey_join(test_df_1, test_df_2, 'l_id', 'r_id', mks[2], 1)

        assert_df_equality(intended_df,result_df, ignore_row_order=True)

#####################################################################


class TestExtractMkVariables():

    def test_expected(self,spark):

        test_df_l = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        test_df_r = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        mks = [
            [test_df_l['first_name'] == test_df_r['first_name'],
             test_df_l['last_name'] == test_df_r['last_name']],

            [F.substring(test_df_l['first_name'], 1, 1) == F.substring(test_df_r['first_name'],
                                                                       1, 1),
                test_df_l['last_name'] == test_df_r['last_name']],

            [F.substring(test_df_l['first_name'], 1, 1) == F.substring(test_df_r['first_name'],
                                                                       1, 1),
                F.substring(test_df_l['last_name'], 1, 1) == F.substring(test_df_r['last_name'],
                                                                         1, 1)]
        ]

        intended_list = sorted(['first_name','last_name'])

        result_list = sorted(extract_mk_variables(test_df_l, mks))

        assert result_list == intended_list

#############################################################################

class TestMkDropna(object):
    def test_expected(self, spark):

        test_df_l = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', '2', None, None, None, '6', '7', '8'],
                "first_name": ['aa', None, 'ab', 'bb', 'aa', 'ax', 'cr', 'cd'],
                "last_name": ['fr', 'gr', None, 'ga', 'gx', 'mx', 'ra', 'ga']
            })))

        test_df_r = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', None, None, None, '8'],
                "first_name": ['ax', None, 'ad', 'bd', 'ar', 'ax', 'cr', 'cd'],
                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', None]
            })))

        mks = [
            [test_df_l['first_name'] == test_df_r['first_name'],
             test_df_l['last_name'] == test_df_r['last_name']],

            [F.substring(test_df_l['first_name'], 1, 1) == F.substring(test_df_r['first_name'],
                                                                       1, 1),
                test_df_l['last_name'] == test_df_r['last_name']],

            [F.substring(test_df_l['first_name'], 1, 1) == F.substring(test_df_r['first_name'],
                                                                       1, 1),
                F.substring(test_df_l['last_name'], 1, 1) == F.substring(test_df_r['last_name'],
                                                                         1, 1)]
        ]

        result_df = mk_dropna(df=test_df_l, match_key=mks)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', None, None, '6', '7', '8'],
                "first_name": ['aa', 'bb', 'aa', 'ax', 'cr', 'cd'],
                "last_name": ['fr', 'ga', 'gx', 'mx', 'ra', 'ga']
            })))

        assert_df_equality(intended_df,result_df, ignore_row_order=True)

#############################################################################

class TestClericalSample(object):
    def test_expected(self, spark):
        df_l = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', '2', '3', '4', '5', '6', '7', '8'],
                "l_first_name": ['aa', None, 'ab', 'bb', 'aa', 'ax', 'cr', 'cd'],
                "l_last_name": ['fr', 'gr', None, 'ga', 'gx', 'mx', 'ra', 'ga']
            })))

        df_r = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', '5', '6', '7', '8'],
                "r_first_name": ['ax', None, 'ad', 'bd', 'ar', 'ax', 'cr', 'cd'],
                "r_last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', None]
            })))

        mks = [
            [df_l['l_first_name'] == df_r['r_first_name'],
             df_l['l_last_name'] == df_r['r_last_name']],

            [F.substring(df_l['l_first_name'], 1, 1) == F.substring(df_r['r_first_name'],
                                                                       1, 1),
                df_l['l_last_name'] == df_r['r_last_name']],

            [F.substring(df_l['l_first_name'], 1, 1) == F.substring(df_r['r_first_name'],
                                                                       1, 1),
                F.substring(df_l['l_last_name'], 1, 1) == F.substring(df_r['r_last_name'],
                                                                         1, 1)]
        ]

        linked_ids = deterministic_linkage(df_l, df_r, "l_id", "r_id", mks, None)

        mk_df = matchkey_dataframe(mks)

        result = clerical_sample(linked_ids, mk_df, df_l, df_r, "l_id", "r_id", n_ids=3)
        result_agg = result.groupby("matchkey").count()

        intended_schema = StructType([
            StructField("matchkey",IntegerType(),False),
            StructField("count",LongType(),False)
        ])

        intended_data = [[1, 2],
                         [2, 3]]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(result_agg, intended_df, ignore_row_order=True)

#############################################################################

class TestDeterministicLinkage(object):
    def test_deterministic_linkage(self, spark):
        df_l = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        df_r = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                        '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        mks = [
            [df_l['first_name'] == df_r['first_name'],
             df_l['last_name'] == df_r['last_name']],

            [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
                df_l['last_name'] == df_r['last_name']],

            [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
                F.substring(df_l['last_name'], 1, 1) == F.substring(df_r['last_name'], 1, 1)]
        ]

        result_df = deterministic_linkage(df_l, df_r, 'l_id', 'r_id', mks, out_dir=None)


        intended_schema = StructType([
            StructField("l_id",StringType(),True),
            StructField("r_id",StringType(),True),
            StructField("matchkey",IntegerType(),False)
        ])

        intended_data = [
          ['10', '10', 1],
          ['11', '11', 1],
          ['12', '12', 1],
          ['13', '13', 1],
          ['14', '14', 1],
          ['15', '15', 1],
          ['16', '16', 1],
          ['17', '17', 1],
          ['18', '18', 1],
          ['19', '19', 1],
          ['20', '20', 1],
          ['6', '6', 1],
          ['7', '7', 1],
          ['8', '8', 1],
          ['9', '9', 1],
          ['1', '1', 2],
          ['2', '2', 2],
          ['3', '3', 2],
          ['4', '4', 2],
          ['5', '5', 2]
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(result_df, intended_df, ignore_row_order=True,ignore_column_order=True)

###################################################################


class TestAssertUniqueMatches():

    def test_expected(self,spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "id_l": ['1', '2', '3', '4', '5'],
                "id_r": ['a', 'b', 'c', 'd', 'e'],
            })))

        intended_df = None

        result_df = assert_unique_matches(test_df, 'id_l', 'id_r')

        assert result_df == intended_df

        x = 0
        try:
            assert_unique_matches(test_df, 'id_l', 'id_r')
        except:
            x = 1
        assert x == 0

        df = spark.createDataFrame(
            (pd.DataFrame({
                "id_l": ['1', '1', '3', '4', '5'],
                "id_r": ['a', 'b', 'c', 'd', 'd'],
            })))

        x = 0
        try:
            assert_unique_matches(df, 'id_l', 'id_r')
        except:
            x = 1
        assert x == 1

###############################################################


class TestMatchkeyCounts(object):

    def test_expected(self,spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "matchkey": ['1', '1', '3', '4', '4'],
                "id_r": ['a', 'b', 'c', 'd', 'e'],
            }))).select("matchkey","id_r")

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "matchkey": ['1', '4', '3'],
                "count": [2,2,1]
            }))).select("matchkey","count")

        result_df = (matchkey_counts(test_df)
                     )

        assert_df_equality(intended_df,result_df,ignore_nullable=True,ignore_row_order=True)

###############################################################


class TestMatchkeyDataframe(object):

    def test_expected(self,spark):

        df_l = spark.createDataFrame(
            (pd.DataFrame({
                "first_name": ['test'] * 10,
                "last_name": ['test'] * 10,
                "uprn": ['test'] * 10,
                "date_of_birth": ['test'] * 10,
            })))

        df_r = spark.createDataFrame(
            (pd.DataFrame({
                "first_name": ['test'] * 10,
                "last_name": ['test'] * 10,
                "uprn": ['test'] * 10,
                "date_of_birth": ['test'] * 10,
            })))

        mks = [
            [
                df_l['first_name'] == df_r['first_name'],
                df_l['last_name'] == df_r['last_name'],
                df_l['uprn'] == df_r['uprn'],
                df_l['date_of_birth'] == df_r['date_of_birth'],
            ],
            [
                F.substring(df_l['first_name'], 0, 2) == F.substring(
                    df_r['first_name'], 0, 2),
                F.substring(df_l['last_name'], 0, 2) == F.substring(
                    df_r['last_name'], 0, 2),
                df_l['uprn'] == df_r['uprn'],
                df_l['date_of_birth'] == df_r['date_of_birth'],
            ]
        ]
        intended_schema = StructType([
            StructField("matchkey",LongType(),True),
            StructField("description",StringType(),True)
        ])

        intended_data = [[1, "[(first_name=first_name),(last_name=last_name),(uprn=uprn),"
                          "(date_of_birth=date_of_birth)]"],
                         [2, "[(substring(first_name,0,2)=substring(first_name,0,2)),"
                          "(substring(last_name,0,2)=substring(last_name,0,2)),"
                          "(uprn=uprn),(date_of_birth=date_of_birth)]"]]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = matchkey_dataframe(mks)

        assert_df_equality(intended_df,result_df, ignore_row_order=True)

###############################################################


class Test_alphaname(object):

    # Test 1
    def test_expected(self,spark):

        test_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
        ])
        test_data = [
            [1, "Homer"],
            [2, "Marge"],
            [3, "Bart"],
            [4, "Lisa"],
            [5, "Maggie"],
        ]

        test_df = spark.createDataFrame(test_data, test_schema)

        intended_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
            StructField("alphaname", StringType(), True),
        ])

        intended_data = [
            [1, "Homer", "EHMOR"],
            [2, "Marge","AEGMR"],
            [3, "Bart","ABRT"],
            [4, "Lisa","AILS"],
            [5, "Maggie","AEGGIM"],
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = alpha_name(test_df,'Forename','alphaname')

        assert_df_equality(intended_df,result_df, ignore_row_order=True)

    # Test 2
    def test_expected_2(self,spark):

        test_schema2 = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Name", StringType(), True),
        ])

        test_data2 = [
            [1, "Romer, Bogdan"],
            [2, "Margarine"],
            [3, None],
            [4, "Nisa"],
            [5, "Moggie"],
        ]

        test_df2 = spark.createDataFrame(test_data2, test_schema2)

        intended_schema2 = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Name", StringType(), True),
            StructField("alphaname", StringType(), True),
        ])  # Note alphaname is always returned as nullable=false

        intended_data2 = [
            [1, "Romer, Bogdan", " ,ABDEGMNOORR"],
            [2, "Margarine","AAEGIMNRR"],
            [3, None, None],
            [4, "Nisa","AINS"],
            [5, "Moggie","EGGIMO"],
        ]

        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)

        result_df2 = alpha_name(test_df2,'Name','alphaname')

        assert_df_equality(intended_df2,result_df2, ignore_row_order=True)


###############################################################

class TestMetaphone(object):

    def test_expected(self,spark):

        test_schema = StructType([
          StructField("ID", IntegerType(), True),
          StructField("Forename", StringType(), True),
        ])
        test_data = [
          [1, "David"],
          [2, "Idrissa"],
          [3, "Edward"],
          [4, "Gordon"],
          [5, "Emma"],
        ]

        test_df = spark.createDataFrame(test_data, test_schema)
        result_df = metaphone(test_df,'Forename','metaname')

        intended_schema = StructType([
          StructField("ID", IntegerType(), True),
          StructField("Forename", StringType(), True),
          StructField("metaname", StringType(), True),
        ])

        intended_data = [
          [1, "David", "TFT"],
          [2, "Idrissa","ITRS"],
          [3, "Edward","ETWRT"],
          [4, "Gordon","KRTN"],
          [5, "Emma","EM"],
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(intended_df,result_df, ignore_row_order=True)


###############################################################

class Test_soundex(object):

    # Test 1
    def test_expected(self,spark):

        test_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
        ])
        test_data = [
            [1, "Homer"],
            [2, "Marge"],
            [3, "Bart"],
            [4, "Lisa"],
            [5, "Maggie"],
        ]

        test_df = spark.createDataFrame(test_data, test_schema)

        result_df = soundex(test_df,'Forename','forename_soundex')

        intended_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
            StructField("forename_soundex", StringType(), True),
        ])

        intended_data = [
            [1, "Homer",'H560'],
            [2, "Marge", 'M620'],
            [3, "Bart", 'B630'],
            [4, "Lisa", 'L200'],
            [5, "Maggie", 'M200'],
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(intended_df,result_df, ignore_row_order=True)

    # Test 2

    def test_expected_2(self,spark):

        test_schema2 = StructType([
            StructField("Surname", StringType(), True),
        ])

        test_data2 = [
            ["McDonald"],
            [None],
            ["MacDonald"],
            ["MacDougall"],
        ]

        test_df2 = spark.createDataFrame(test_data2, test_schema2)

        result_df2 = soundex(test_df2,'Surname','soundex')

        intended_schema2 = StructType([
            StructField("Surname", StringType(), True),
            StructField("soundex", StringType(), True),
        ])

        intended_data2 = [
            ["McDonald",'M235'],
            [None, None],
            ["MacDonald",'M235'],
            ["MacDougall",'M232'],
        ]

        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)

        assert_df_equality(intended_df2,result_df2, ignore_row_order=True)

###############################################################

class Test_std_lev_score(object):

    # Test 1

    def test_expected(self,spark):

        test_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
            StructField("Forename_2", StringType(), True),
        ])
        test_data = [
            [1, "Homer",'Milhouse'],
            [2, "Marge",'Milhouse'],
            [3, "Bart",'Milhouse'],
            [4, "Lisa",'Milhouse'],
            [5, "Maggie",'Milhouse'],
        ]

        test_df = spark.createDataFrame(test_data, test_schema)

        result_df = test_df.withColumn('forename_lev',
                                       std_lev_score(F.col('Forename'), F.col('Forename_2')))

        intended_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
            StructField("Forename_2", StringType(), True),
            StructField("forename_lev", DoubleType(), True),
        ])
        intended_data = [
            [1, "Homer",'Milhouse', 1 / 8],
            [2, "Marge",'Milhouse', 2 / 8],
            [3, "Bart",'Milhouse', 0 / 8],
            [4, "Lisa",'Milhouse', 2 / 8],
            [5, "Maggie",'Milhouse', 2 / 8],
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(intended_df,result_df, ignore_row_order=True)

    # Test 2

    def test_expected_2(self,spark):

        test_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
            StructField("Forename_2", StringType(), True),
        ])

        test_data2 = [
            [1, "Homer",'Milhouse'],
            [2, "Marge",'Milhouse'],
            [3, "Bart",'Milhouse'],
            [4, "Lisa",'Milhouse'],
            [5, "Maggie",'Milhouse'],
            [6, None,'Milhouse'],
            [7, 'Milhouse', None],
            [8, 'Milhouse','Milhouse'],
        ]

        test_df2 = spark.createDataFrame(test_data2, test_schema)

        result_df2 = test_df2.withColumn('forename_lev',
                                         std_lev_score(F.col('Forename'), F.col('Forename_2')))

        intended_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("Forename", StringType(), True),
            StructField("Forename_2", StringType(), True),
            StructField("forename_lev", DoubleType(), True),
        ])

        intended_data2 = [
            [1, "Homer",'Milhouse', 1 / 8],
            [2, "Marge",'Milhouse', 2 / 8],
            [3, "Bart",'Milhouse', 0 / 8],
            [4, "Lisa",'Milhouse', 2 / 8],
            [5, "Maggie",'Milhouse', 2 / 8],
            [6, None,'Milhouse', None],
            [7, 'Milhouse',None, None],
            [8, "Milhouse",'Milhouse', 1 / 1],
        ]

        intended_df2 = spark.createDataFrame(intended_data2, intended_schema)

        assert_df_equality(intended_df2,result_df2, ignore_row_order=True)

###############################################################

class TestJaro(object):
    def test_expected(self,spark):
        test_schema = StructType([
          StructField("string1", StringType(), True),
          StructField("string2", StringType(), True)
      ])

        test_data = [
          ["Hello", "HHheello"],
          ["Hello", " h e l l o"],
          ["Hello", "olleH"],
          ["Hello", "H1234"],
          ["Hello", "1234"]
        ]

        test_df = spark.createDataFrame(test_data, test_schema)
        result = test_df.withColumn("jaro", jaro(test_df["string1"], test_df["string2"]))
        assert sorted(
          result.toPandas().loc[:, "jaro"].tolist()
        ) == sorted(
          [0.875, 0.6333333253860474, 0.6000000238418579, 0.46666666865348816, 0.0]
        )

###############################################################

class TestJaroWinkler(object):
    def test_expected(self,spark):
        test_schema = StructType([
          StructField("string1", StringType(), True),
          StructField("string2", StringType(), True)
        ])

        test_data = [
          ["Hello", "HHheello"],
          ["Hello", " h e l l o"],
          ["Hello", "olleH"],
          ["Hello", "H1234"],
          ["Hello", "1234"]
        ]

        test_df = spark.createDataFrame(test_data, test_schema)
        result = test_df.withColumn("jaro_winkler", jaro_winkler(test_df["string1"], test_df["string2"]))
        assert sorted(
          result.toPandas().loc[:, "jaro_winkler"].tolist()
        ) == sorted(
          [0.887499988079071,
           0.6333333253860474,
           0.6000000238418579,
           0.46666666865348816,
           0.0
          ]
        )

###############################################################

class TestDifflibSequenceMatcher(object):
    def test_expected(self,spark):
        test_schema = StructType([
          StructField("string1", StringType(), True),
          StructField("string2", StringType(), True)
        ])

        test_data = [
          ["David", "Emily"],
          ["Idrissa", "Emily"],
          ["Edward", "Emily"],
          ["Gordon", "Emily"],
          ["Emma", "Emily"]
        ]

        test_df = spark.createDataFrame(test_data, test_schema)
        result_df = test_df.withColumn(
          "difflib",
          difflib_sequence_matcher(
            F.col("string1"), F.col("string2")
          )
        )

        intended_schema = StructType([
          StructField("string1", StringType(), True),
          StructField("string2", StringType(), True),
          StructField("difflib", FloatType(), True)
        ])

        intended_data = [
          ["David", "Emily", 0.2],
          ["Idrissa", "Emily", 0.16666667],
          ["Edward", "Emily", 0.18181819],
          ["Gordon", "Emily", 0.0],
          ["Emma", "Emily", 0.44444445]
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assert_df_equality(intended_df,result_df, ignore_row_order=True)

###############################################################

class TestBlocking(object):
    def test_expected(self, spark):
        test_schema = StructType([
          StructField("ID_1", IntegerType(), True),
          StructField("age_df1", IntegerType(), True),
          StructField("sex_df1", StringType(), True),
          StructField("pc_df1", StringType(), True)
        ])

        test_data = [
          [1, 1, "Male", "gu1111"],
          [2, 1, "Female", "gu1211"],
          [3, 56, "Male", "gu2111"],
        ]
        df1 = spark.createDataFrame(test_data, test_schema)

        test_schema = StructType([
          StructField("ID_2", IntegerType(), True),
          StructField("age_df2", IntegerType(), True),
          StructField("sex_df2", StringType(), True),
          StructField("pc_df2", StringType(), True)
        ])

        test_data = [
          [6, 2, "Female", "gu1211"],
          [5, 56, "Male", "gu1411"],
          [4, 7, "Female", "gu1111"],
        ]
        df2 = spark.createDataFrame(test_data, test_schema)

        id_vars = ['ID_1', 'ID_2']
        blocks = {'pc_df1': 'pc_df2'}
        result_df = blocking(df1, df2, blocks, id_vars)

        expected_schema = StructType([
          StructField("ID_1", IntegerType(), True),
          StructField("age_df1", IntegerType(), True),
          StructField("sex_df1", StringType(), True),
          StructField("pc_df1", StringType(), True),
          StructField("ID_2", IntegerType(), True),
          StructField("age_df2", IntegerType(), True),
          StructField("sex_df2", StringType(), True),
          StructField("pc_df2", StringType(), True)
        ])

        expected_data = [
          [1, 1, "Male", "gu1111", 4, 7, "Female", "gu1111"],
          [2, 1, "Female", "gu1211", 6, 2, "Female", "gu1211"]
        ]
        expected_df = spark.createDataFrame(expected_data, expected_schema)

        assert_df_equality(expected_df, result_df, ignore_row_order=True)

###############################################################

class TestAssertUnique(object):

    def test_expected(self, spark):
        test_schema = StructType([
          StructField("colA", IntegerType(), True),
          StructField("colB", IntegerType(), True)
        ])

        test_data = [
          [1, 1],
          [1, 2]
        ]
        df = spark.createDataFrame(test_data, test_schema)
        try:
          assert_unique(df, "colA")
        except AssertionError:
          pass
        assert_unique(df, ["colB"])

class TestClusterNumber(object):
    def test_expected(self, spark):
        """
        If this test fails because of the graphframes package not being found,
        make sure you have both graphframes and graphframes_wrapper installed
        via pip3.
        """

        test_schema = StructType([
          StructField("id1", StringType(), True),
          StructField("id2", StringType(), True)
        ])

        test_data = [
          ["1a", "2b"],
          ["3a", "3b"],
          ["2a", "1b"],
          ["3a", "7b"],
          ["1a", "8b"],
          ["2a", "9b"]
        ]
        df = spark.createDataFrame(test_data, test_schema)
        result_df = cluster_number(df, id_1 = "id1", id_2 = "id2")
        assert result_df is not None

        intended_schema = StructType([
          StructField("id1", StringType(), True),
          StructField("id2", StringType(), True),
          StructField("Cluster_Number", IntegerType(), True),
        ])

        intended_data = [
          ["2a", "1b", 1],
          ["2a", "9b", 1],
          ["3a", "3b", 2],
          ["3a", "7b", 2],
          ["1a", "8b", 3],
          ["1a", "2b", 3]
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(intended_df, result_df, ignore_row_order=True)
