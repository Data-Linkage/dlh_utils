'''
Pytesting on Linkage functions
'''

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,IntegerType
import pandas as pd
import pytest
import chispa
from chispa import assert_df_equality
from dlh_utils.linkage import order_matchkeys,matchkey_join,extract_mk_variables,\
demographics,demographics_compare,assert_unique_matches,matchkey_counts,\
matchkey_dataframe

pytestmark = pytest.mark.usefixtures("spark")

#############################################################################

class TestOrderMatchkeys(object):

    def test_expected(self,spark):

        dfo = spark.createDataFrame(
            (pd.DataFrame({
                "uprn": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                         '14', '15','16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
          })))

        dffn = spark.createDataFrame(
            (pd.DataFrame({
                "uprn": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
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

        testDf = pd.DataFrame({
            'mks': mks,
            'count': [(dfo.join(dffn, on=mk, how='inner')).count()
                    for mk in mks]
      })

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
                "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        test_df_2 = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        mks = [
            [test_df_1['first_name'] == test_df_2['first_name'],
             test_df_1['last_name'] == test_df_2['last_name']],

            [F.substring(test_df_1['first_name'], 1, 1) == F.substring(test_df_2['first_name'],\
                                                                       1, 1),
                test_df_1['last_name'] == test_df_2['last_name']],

            [F.substring(test_df_1['first_name'], 1, 1) == F.substring(test_df_2['first_name'],\
                                                                       1, 1),
                F.substring(test_df_1['last_name'], 1, 1) == F.substring(test_df_2['last_name'],\
                                                                         1, 1)]
        ]

        intended_schema = StructType([
                    StructField("l_id",StringType(),True),
                    StructField("r_id",StringType(),True),
                    StructField("matchkey",IntegerType(),False)
        ])

        intended_data =[['7', '7', 1],
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

        assert_df_equality(intended_df,result_df)

#####################################################################

class TestExtractMkVariables():

    def test_expected(self,spark):

        test_df_l = spark.createDataFrame(
            (pd.DataFrame({
                "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        test_df_r = spark.createDataFrame(
            (pd.DataFrame({
                "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
                         '14', '15', '16', '17', '18', '19', '20'],

                "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                               'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

                "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                              'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
            })))

        mks = [
            [test_df_l['first_name'] == test_df_r['first_name'],
             test_df_l['last_name'] == test_df_r['last_name']],

            [F.substring(test_df_l['first_name'], 1, 1) == F.substring(test_df_r['first_name'],\
                                                                       1, 1),
                test_df_l['last_name'] == test_df_r['last_name']],

            [F.substring(test_df_l['first_name'], 1, 1) == F.substring(test_df_r['first_name'],\
                                                                       1, 1),
                F.substring(test_df_l['last_name'], 1, 1) == F.substring(test_df_r['last_name'],\
                                                                         1, 1)]
        ]

        intended_list = sorted(['first_name','last_name'])

        result_list = sorted(extract_mk_variables(test_df_l, mks))

        assert result_list == intended_list

#############################################################################
#
# unable to do pytest on the following code as function 'deterministic_linkage()'
# is missing the argument 'out_dir'
#
#def test_deterministic_linkage():
#   spark = SparkSession.builder.getOrCreate()
#    df_l = spark.createDataFrame(
#        (pd.DataFrame({
#            "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
#                     '14', '15', '16', '17', '18', '19', '20'],
#
#            "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
#                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],
#
#            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
#                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
#        })))
#
#    df_r = spark.createDataFrame(
#        (pd.DataFrame({
#            "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',\
#                    '14', '15', '16', '17', '18', '19', '20'],
#
#            "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
#                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],
#
#            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
#                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
#        })))
#
#    mks = [
#        [df_l['first_name'] == df_r['first_name'],
#         df_l['last_name'] == df_r['last_name']],
#
#        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
#            df_l['last_name'] == df_r['last_name']],
#
#        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
#            F.substring(df_l['last_name'], 1, 1) == F.substring(df_r['last_name'], 1, 1)]
#    ]
#
#    result_df = deterministic_linkage(df_l, df_r, 'l_id', 'r_id', mks).filter(F.col('l_id') <= 5)\
#                                                                      .where(F.col('matchkey')\
#                                                                       == 1)
#                                                                      .count()==5
#
#    assert ((deterministic_linkage(df_l, df_r, 'l_id', 'r_id', mks)
#             .filter(F.col('l_id') <= 5))
#            .where(F.col('matchkey') == 1)
#            .count() == 5)
#
#    assert ((deterministic_linkage(df_l, df_r, 'l_id', 'r_id', mks)
#             .filter(F.col('l_id') > 5))
#            .where(F.col('matchkey') == 0)
#            .count() == 15)
#
###################################################################

class TestDemographics():

    def test_expected(self,spark):

        test_df_raw = spark.createDataFrame(
            (pd.DataFrame({
                "id": [x for x in range(40)],
                "sex": (['M']*20)+(['F']*20),
                "age_group": (['10-20']*10)+(['20-30']*10)+(['30-40']*10)+(['50-60']*10),
            }))).select('id','sex','age_group')

        intended_data_raw = spark.createDataFrame(
            (pd.DataFrame({
                'variable': ['age_group','age_group','age_group','age_group','sex','sex'],
                'value': ['10-20','20-30','30-40','50-60','F','M'],
                'count': [10,10,10,10,20,20],
                'total_count': [40,40,40,40,40,40]
            }))).select('variable','value','count','total_count')

        result_df_raw = demographics(*['sex', 'age_group'],
                               df=test_df_raw, identifier='id')

        assert_df_equality(intended_data_raw,result_df_raw,ignore_nullable = True,\
                           ignore_schema = True)

        test_df_linked = spark.createDataFrame(
            (pd.DataFrame({
                "id": [x for x in range(20)],
                "sex": (['M']*10)+(['F']*10),
                "age_group": (['10-20']*5)+(['20-30']*5)+(['30-40']*5)+(['50-60']*5),
            }))).select('id','sex','age_group')

        intended_data_linked = spark.createDataFrame(
            (pd.DataFrame({
                'variable': ['age_group','age_group','age_group','age_group','sex','sex'],
                'value': ['10-20','20-30','30-40','50-60','F','M'],
                'count': [5,5,5,5,10,10],
                'total_count': [20,20,20,20,20,20]
            }))).select('variable','value','count','total_count')

        result_df_linked = demographics(*['sex', 'age_group'],
                                  df=test_df_linked, identifier='id')

        assert_df_equality(intended_data_linked,result_df_linked,ignore_nullable = True,\
                           ignore_schema = True)

####################################################################

class TestDemographicsCompare():

    def test_expected(self,spark):

        spark = SparkSession.builder.getOrCreate()

        df_raw = spark.createDataFrame(
            (pd.DataFrame({
                "id": [x for x in range(40)],
                "sex": (['M']*20)+(['F']*20),
                "age_group": (['10-20']*10)+(['20-30']*10)+(['30-40']*10)+(['50-60']*10),
            })))

        df_linked = spark.createDataFrame(
            (pd.DataFrame({
                "id": [x for x in range(20)],
                "sex": (['M']*10)+(['F']*10),
                "age_group": (['10-20']*5)+(['20-30']*5)+(['30-40']*5)+(['50-60']*5),
            })))

        test_raw = demographics(*['sex', 'age_group'],
                               df=df_raw, identifier='id')

        test_linked = demographics(*['sex', 'age_group'],
                                  df=df_linked, identifier='id')

        intended_data = spark.createDataFrame(
            (pd.DataFrame({
                'variable': ['age_group','age_group','age_group','age_group','sex','sex'],
                'value': ['10-20','20-30','30-40','50-60','F','M'],
                'count_raw': [10,10,10,10,20,20],
                'total_count_raw': [40,40,40,40,40,40],
                'count_linked': [5,5,5,5,10,10],
                'total_count_linked': [20,20,20,20,20,20],
                'match_rate': [0.5,0.5,0.5,0.5,0.5,0.5],
                'proportional_count': [5,5,5,5,10,10],
                'proportional_discrepency': [0.0,0.0,0.0,0.0,0.0,0.0]
            }))).select('variable','value','count_raw','total_count_raw','count_linked',
                        'total_count_linked','match_rate','proportional_count',
                        'proportional_discrepency')

        result_df = demographics_compare(test_raw, test_linked)

        assert_df_equality(intended_data,result_df,ignore_nullable = True,ignore_schema = True)

        df_raw = spark.createDataFrame(
            (pd.DataFrame({
                "id": [x for x in range(40)],
                "sex": (['M']*20)+(['F']*20),
                "age_group": (['10-20']*10)+(['20-30']*10)+(['30-40']*10)+(['50-60']*10),
            })))

        df_linked = spark.createDataFrame(
            (pd.DataFrame({
                "id": [x for x in range(20)],
                "sex": (['M']*15)+(['F']*5),
                "age_group": (['10-20']*5)+(['20-30']*5)+(['30-40']*5)+(['50-60']*5),
            })))

        test_dem_raw1 = demographics(*['sex', 'age_group'],
                               df=df_raw, identifier='id')

        test_dem_linked1 = demographics(*['sex', 'age_group'],
                               df=df_linked, identifier='id')

        intended_data1 = spark.createDataFrame(
            (pd.DataFrame({
                'variable': ['age_group','age_group','age_group','age_group','sex','sex'],
                'value': ['10-20','20-30','30-40','50-60','F','M'],
                'count_raw': [10,10,10,10,20,20],
                'total_count_raw': [40,40,40,40,40,40],
                'count_linked': [5,5,5,5,5,15],
                'total_count_linked': [20,20,20,20,20,20],
                'match_rate': [0.5,0.5,0.5,0.5,0.5,0.5],
                'proportional_count': [5,5,5,5,10,10],
                'proportional_discrepency': [0.0,0.0,0.0,0.0,-0.5,0.5]
            }))).select('variable','value','count_raw','total_count_raw','count_linked',
                        'total_count_linked','match_rate','proportional_count',
                        'proportional_discrepency')


        result_df1 = demographics_compare(test_dem_raw1, test_dem_linked1)

        assert_df_equality(intended_data1,result_df1,ignore_nullable = True,ignore_schema = True)

####################################################################

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

        assert_df_equality(intended_df,result_df, ignore_nullable = True, ignore_row_order = True)

###############################################################

class TestMatchkeyDataframe(object):

    def test_expected(self,spark):

        df_l = spark.createDataFrame(
            (pd.DataFrame({
                "first_name": ['test']*10,
                "last_name": ['test']*10,
                "uprn": ['test']*10,
                "date_of_birth": ['test']*10,
            })))

        df_r = spark.createDataFrame(
            (pd.DataFrame({
                "first_name": ['test']*10,
                "last_name": ['test']*10,
                "uprn": ['test']*10,
                "date_of_birth": ['test']*10,
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

        intended_data = [[1, "[(first_name=first_name),(last_name=last_name),(uprn=uprn),"\
                          "(date_of_birth=date_of_birth)]"],\
                         [2, "[(substring(first_name,0,2)=substring(first_name,0,2)),"\
                          "(substring(last_name,0,2)=substring(last_name,0,2)),"
                          "(uprn=uprn),(date_of_birth=date_of_birth)]"]]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = matchkey_dataframe(mks)

        assert_df_equality(intended_df,result_df)


# unable to do pytest on the following code as function 'deterministic_linkage()'
# is missing the argument 'out_dir'
#
#    df_l = spark.createDataFrame(
#        (pd.DataFrame({
#            "id_l": [-1, -2, -3],
#            "first_name": ['AMY', 'AMY', 'AMY'],
#            "last_name": ['SMITH', 'SMITH', 'SMITH'],
#            "date_of_birth": ['a', None, 'b'],
#            "uprn": ['a', 'b', None],
#            "sex": ['F', 'F', 'F']
#        })))
#
#    df_r = spark.createDataFrame(
#        (pd.DataFrame({
#            "id_r": [-1, -2, -3],
#            "first_name": ['AMY', 'AMY', 'AMY'],
#            "last_name": ['SMITH', 'SMITH', 'SMITH'],
#            "date_of_birth": ['a', None, 'b'],
#            "uprn": ['a', 'b', None],
#            "sex": ['F', 'F', 'F']
#        })))
#
#    mks = [
#        [
#            df_l['first_name'] == df_r['first_name'],
#            df_l['last_name'] == df_r['last_name'],
#            df_l['sex'] == df_r['sex'],
#            df_l['uprn'] == df_r['uprn'],
#            df_l['date_of_birth'] == df_r['date_of_birth'],
#        ],
#        [
#            df_l['first_name'] == df_r['first_name'],
#            df_l['last_name'] == df_r['last_name'],
#            df_l['sex'] == df_r['sex'],
#            df_l['uprn'] == df_r['uprn'],
#        ],
#        [
#            df_l['first_name'] == df_r['first_name'],
#            df_l['last_name'] == df_r['last_name'],
#            df_l['sex'] == df_r['sex'],
#            df_l['date_of_birth'] == df_r['date_of_birth'],
#        ],
#    ]
#
#    assert (li.deterministic_linkage(df_l, df_r, 'id_l', 'id_r', mks)
#            .where(F.col('id_l') != F.col('id_r'))).count() == 0
