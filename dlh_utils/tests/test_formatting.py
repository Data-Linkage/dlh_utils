'''
Pytesting on Formatting functions.
'''
import pandas as pd
import dlh_utils.formatting

#############################################################################


class TestExportToExcel:
    """Test of export to excel function."""
    def test_default_single_dataframe(self):
        """
        The simplest case: exporting one dataframe to a single-sheet workbook
        without any formatting.
        """
        df = pd.DataFrame(
            {
                "firstname": ["Alan", "Claire", "Josh", "Bob"],
                "lastname": ["Jones", "Llewelyn", "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, 0, 4.3],
                "numeric_B": [-500, 221, 1, 0],
            }
        )
        wb = dlh_utils.formatting.export_to_excel({"Sheet1": df}, local_path="/tmp/pytest.xlsx")

        assert len(wb.worksheets) == 1
        for i, c in enumerate(df.columns):
            assert (wb["Sheet1"].cell(1, i + 1).value) == c
        for rownum in range(df.shape[0]):
            for i, c in enumerate(df.columns):
                a = wb["Sheet1"].cell(rownum + 2, i + 1).value
                b = df.loc[rownum, c]
                assert a == b

    def test_list_of_columns_single_dataframe(self):
        """
        The simplest case: exporting one dataframe to a single-sheet workbook
        without any formatting.
        """
        df = pd.DataFrame(
            {
                "firstname": ["Alan", "Claire", "Josh", "Bob"],
                "lastname": ["Jones", "Llewelyn", "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, 0, 4.3],
                "numeric_B": [-500, 221, 1, 0],
            }
        )
        wb = dlh_utils.formatting.export_to_excel(
          {"Sheet1": df},
          columns = ["firstname", "numeric_A"],
          local_path="/tmp/pytest.xlsx"
        )

        assert len(wb.worksheets) == 1
        for i, c in enumerate(["firstname", "numeric_A"]):
            assert (wb["Sheet1"].cell(1, i + 1).value) == c
        for rownum in range(df.shape[0]):
            for i, c in enumerate(["firstname", "numeric_A"]):
                a = wb["Sheet1"].cell(rownum + 2, i + 1).value
                b = df.loc[rownum, c]
                assert a == b

    def test_default_two_dataframes(self):
        """
        Exporting two dataframes to a workbook, each on its own worksheet
        without any formatting.
        """
        dfA = pd.DataFrame(
            {
                "firstname": ["Alan", "Claire", "Josh", "Bob"],
                "lastname": ["Jones", "Llewelyn", "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, 0, 4.3],
                "numeric_B": [-500, 221, 1, 0],
            }
        )
        dfB = pd.DataFrame(
            {
                "firstname": ["Anne", "Betty", "Carlo", "Daphne"],
                "lastname": ["Abbot", "Benson", "Carruthers", "De Morgan"],
                "numeric_A": [1, 2, 3, 4],
                "numeric_B": [91, 92, 93, 94],
            }
        )
        wb = dlh_utils.formatting.export_to_excel(
            {
                "Dataframe A": dfA,
                "Dataframe B": dfB
            },
            local_path="/tmp/pytest.xlsx"
        )
        assert len(wb.worksheets) == 2
        for i, c in enumerate(dfA.columns):
            assert (wb["Dataframe A"].cell(1, i + 1).value) == c
        for rownum in range(dfA.shape[0]):
            for i, c in enumerate(dfA.columns):
                a = wb["Dataframe A"].cell(rownum + 2, i + 1).value
                b = dfA.loc[rownum, c]
                assert a == b
        for i, c in enumerate(dfB.columns):
            assert (wb["Dataframe B"].cell(1, i + 1).value) == c
        for rownum in range(dfB.shape[0]):
            for i, c in enumerate(dfB.columns):
                a = wb["Dataframe B"].cell(rownum + 2, i + 1).value
                b = dfB.loc[rownum, c]
                assert a == b

    def test_custom_columns(self):
        """
        Exporting one dataframe to a single-sheet workbook
        without any formatting but specifying which columns to export.
        """
        df = pd.DataFrame(
            {
                "firstname": ["Anne", "Betty", "Carlo", "Daphne"],
                "lastname": ["Abbot", "Benson", "Carruthers", "De Morgan"],
                "numeric_A": [1, 2, 3, 4],
                "numeric_B": [91, 92, 93, 94],
            }
        )
        column_list = ["numeric_B", "lastname"]
        wb = dlh_utils.formatting.export_to_excel(
            {
                "Dataframe C": df
            },
            local_path="/tmp/pytest.xlsx",
            columns={
                "Dataframe C": column_list
            }
        )
        assert len(wb.worksheets) == 1
        for i, c in enumerate(column_list):
            assert (wb["Dataframe C"].cell(1, i + 1).value) == c
        for rownum in range(df.shape[0]):
            for i, c in enumerate(column_list):
                a = wb["Dataframe C"].cell(rownum + 2, i + 1).value
                b = df.loc[rownum, c]
                assert a == b

    def test_custom_styles(self):
        """
        Exporting one dataframe to a single-sheet workbook
        with basic formatting. More complex formatting is tested
        elsewhere.
        """
        df = pd.DataFrame(
            {
                "firstname": ["Anne", "Betty", "Carlo", "Daphne"],
                "lastname": ["Abbot", "Benson", "Carruthers", "De Morgan"],
                "numeric_A": [-1, 2, -3, 4],
                "numeric_B": [91, 92, 93, 94],
            }
        )
        wb = dlh_utils.formatting.export_to_excel(
            {
                "Dataframe D": df
            },
            local_path="/tmp/pytest.xlsx",
            styles={
                "Dataframe D": {
                    dlh_utils.formatting.style_on_cutoff: "numeric_A"
                }
            }
        )
        assert len(wb.worksheets) == 1
        assert wb["Dataframe D"].cell(2, 3).has_style
        assert wb["Dataframe D"].cell(2, 3).fill.fgColor.rgb == "00FF0000"

#############################################################################


class TestApplyStyles:

    def test_default_single_style_1(self):
        """
        Applying a built-in function (style_on_cutoff) to a single column
        using its default settings.
        """
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        # Default behaviour
        sdf = dlh_utils.formatting.apply_styles(
            df,
            {
                dlh_utils.formatting.style_on_cutoff: "numeric_A"
            }
        )
        style_applied = sdf.export()
        # Only one style was applied
        assert len(style_applied) == 1
        # Style was applied using applymap
        assert "Styler.applymap" in str(style_applied[0][0])
        # Style was style_on_cutoff
        assert style_applied[0][1][0] == dlh_utils.formatting.style_on_cutoff
        # Style was applied to numeric_A only
        assert style_applied[0][1][1] == "numeric_A"


    def test_default_two_styles(self):
        """
        Applying a built-in function (style_on_cutoff) to two columns
        using its default settings.
        """
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        # Default behaviour with multiple columns
        sdf = dlh_utils.formatting.apply_styles(
            df,
            {
                dlh_utils.formatting.style_on_condition: ["numeric_A", "numeric_B"]
            }
        )
        style_applied = sdf.export()
        # Two styles were applied
        assert len(style_applied) == 2
        # Styles were applied using applymap
        assert "Styler.applymap" in str(style_applied[0][0])
        assert "Styler.applymap" in str(style_applied[1][0])
        # Styles were style_on_condition
        assert style_applied[0][1][0] == dlh_utils.formatting.style_on_condition
        assert style_applied[1][1][0] == dlh_utils.formatting.style_on_condition
        # Styles were applied to the appropriate columns
        assert style_applied[0][1][1] == "numeric_A"
        assert style_applied[1][1][1] == "numeric_B"

    def test_partial_function(self):
        """
        Applying a built-in function (style_on_cutoff) to one column,
        setting a custom parameter value.
        """
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        from functools import partial
        f = partial(dlh_utils.formatting.style_on_condition, property='color')
        sdf = dlh_utils.formatting.apply_styles(
            df,
            {
                f: "numeric_A"
            }
        )
        style_applied = sdf.export()
        # Only one style was applied
        assert len(style_applied) == 1
        # Style was applied using applymap
        assert "Styler.applymap" in str(style_applied[0][0])
        # The right function was applied
        assert style_applied[0][1][0] is f
        # Style was applied to numeric_A only
        assert style_applied[0][1][1] == "numeric_A"

    def test_default_custom_function(self):
        """
        Applying a user-defined function to a column.
        """
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        # User-defined function
        def udf(x):
            return "a" in x.lower()

        sdf = dlh_utils.formatting.apply_styles(
            df,
            {
                udf: "lastname"
            }
        )
        style_applied = sdf.export()
        # Only one style was applied
        assert len(style_applied) == 1
        # Style was applied using applymap
        assert "Styler.applymap" in str(style_applied[0][0])
        # The right function was applied
        assert style_applied[0][1][0] is udf
        # Style was applied to lastname only
        assert style_applied[0][1][1] == "lastname"

#############################################################################


class TestStyleOnCutoff:

    def test_default_behaviour(self):
        """
        Testing the function when no optional parameters are passed.
        """
        # Default behaviour
        result = dlh_utils.formatting.style_on_cutoff(5)
        assert result == "background-color : green;"
        result = dlh_utils.formatting.style_on_cutoff(0)
        assert result == "background-color : white;"
        result = dlh_utils.formatting.style_on_cutoff(-5)
        assert result == "background-color : red;"
        result = dlh_utils.formatting.style_on_cutoff("ERROR")
        assert result == "background-color : black;"

    def test_custom_behaviour(self):
        """
        Testing the function when various optional parameters are passed.
        """
        result = dlh_utils.formatting.style_on_cutoff(
            5,
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color"
        )
        assert result == "color : #AABBAA;"
        result = dlh_utils.formatting.style_on_cutoff(
            3,
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color"
        )
        assert result == "color : #33DD33;"
        result = dlh_utils.formatting.style_on_cutoff(
            -5,
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color"
        )
        assert result == "color : #00FF00;"
        result = dlh_utils.formatting.style_on_cutoff(
            "ERROR",
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color"
        )
        assert result == "color : green;"

#############################################################################


class TestStyleOnCondition:

    def test_default_behaviour(self):
        """
        Testing the function when no optional parameters are passed.
        """
        result = dlh_utils.formatting.style_on_condition(0)
        assert result == "font-weight : bold;"
        result = dlh_utils.formatting.style_on_condition(1)
        assert result == "font-weight : normal;"

    def test_custom_property(self):
        """
        Testing the function when no optional formatting parameters are passed.
        """
        result = dlh_utils.formatting.style_on_condition(
            1,
            property="color",
            true_style="'red'",
            false_style="'blue'",
            error_style=None
        )
        assert result == "color : 'blue';"

    def test_custom_condition(self):
        """
        Testing the function when a custom condition is specified.
        """
        result = dlh_utils.formatting.style_on_condition(
            10,
            condition=lambda x:x % 2 == 0
        )
        assert result == "font-weight : bold;"

#############################################################################


class TestStyleColourGradient:

    def test_min_value(self):
        """
        Testing the function when the lowest possible value is passed.
        """
        result = dlh_utils.formatting.style_colour_gradient(
            1, 1, 10,
            min_colour="FF0000",
            max_colour="00DDFF",
            error_colour=None
        )
        assert result == "background-color : #FF0000;"

    def test_max_value(self):
        """
        Testing the function when the highest possible value is passed.
        """
        result = dlh_utils.formatting.style_colour_gradient(
            10, 1, 10,
            min_colour="FF0000",
            max_colour="00DDFF",
            error_colour=None,
            property="color"
        )
        assert result == "color : #00DDFF;"

    def test_intermediate_value(self):
        """
        Testing the function when an intermediate value is passed.
        """
        result = dlh_utils.formatting.style_colour_gradient(
            4, 1, 10,
            min_colour="FF0000",
            max_colour="00DDFF",
            error_colour=None
        )
        assert result == "background-color : #AA4955;"

    def test_error_value(self):
        """
        Testing the function when an invalid value is passed amd error_colour is specified.
        """
        result = dlh_utils.formatting.style_colour_gradient(
            "ERROR", 1, 10,
            min_colour="FF0000",
            max_colour="00DDFF",
            error_colour="AAAAAA"
        )
        assert result == "background-color : #AAAAAA;"

#############################################################################


class TestStyleMapValues:

    def test_numeric(self):
        """
        Testing the function for a numeric input.
        """
        result = dlh_utils.formatting.style_map_values(
            1,
            {0:"black", 1:"red"}
        )
        assert result == "background-color : red;"

    def test_boolean(self):
        """
        Testing the function for a boolean input.
        """
        result = dlh_utils.formatting.style_map_values(
            True,
            {True:"black", False:"red"},
            property="color"
        )
        assert result == "color : black;"

        # NB 1 is truthy in Python, so this is expected behaviour
        result = dlh_utils.formatting.style_map_values(
            1,
            {True:"black", False:"red"},
            property="color"
        )
        assert result == "color : black;"

    def test_default_style(self):
        """
        Testing the default style is used when the value is absent from the
        mapping_dictionary.
        """
        result = dlh_utils.formatting.style_map_values(
            2,
            {0:"black", 1:"red"},
            property="color",
            default_style="green"
        )
        assert result == "color : green;"

    def test_error_style(self):
        """
        Testing the function applies the error style correctly.
        A TypeError arises because [1] is a list, which cannot be used
        to look up a key in a dictionary. This is "swallowed" by style_map_values,
        which instead returns the error_style.
        """
        result = dlh_utils.formatting.style_map_values(
            [1],
            {True:"black", False:"red"},
            property="color",
            error_style="green"
        )
        assert result == "color : green;"
