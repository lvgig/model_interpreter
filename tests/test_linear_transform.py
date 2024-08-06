import pytest  # type: ignore
from model_interpreter.interpreter import ModelInterpreter  # type: ignore
from hypothesis import given, strategies as st, settings, HealthCheck
import pandas as pd  # type: ignore
import json


class TestLinearTransform(object):
    """Tests for ModelInterpreter.transform() when the model is linear."""

    def test_output_transform_is_serializable(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check no exception raised if x.transform() is dumped to json string"""
        try:
            MI.fit(lin_model, lin_X_train)
            contributions = MI.transform(single_row, return_type="dicts")
            json.dumps(contributions)
        except TypeError:
            pytest.fail("Unexpected TypeError: Output of transform is not serilizable")

    # Tests below are for a linear model
    #################################################################################################################

    def test_expected_output_name_value_dicts(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "name_value_dicts" """

        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, return_type="name_value_dicts")
        keys_to_check = {"Name", "Value"}
        check_flag = all(
            isinstance(item, dict) and keys_to_check == set(item.keys())
            for item in df_transformed
        )
        expected = True
        actual = check_flag

        assert expected == actual

    def test_model_expected_output_rounded_decimals(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that the shap value contributions are rounded to the correct precision"""

        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(
            single_row, return_type="name_value_dicts", return_precision=10
        )
        check_flag = all(
            round(item["Value"], 10) == item["Value"] for item in df_transformed
        )
        expected = True
        actual = check_flag

        assert expected == actual

    def test_expected_output_name_value_dicts_contents(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that the dictionaries in the output when transform return_type is "name_value_dicts" contain the expected outputs"""

        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, return_type="name_value_dicts")
        check_flag = all(
            isinstance(item["Name"], str) and isinstance(item["Value"], float)
            for item in df_transformed
        )
        expected = True
        actual = check_flag

        assert expected == actual

    def test_expected_output_tuples(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "tuples" """

        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, return_type="tuples")
        check_flag = all(isinstance(item, tuple) for item in df_transformed)
        expected = True
        actual = check_flag

        assert expected == actual

    def test_expected_output_dicts(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "dicts" """
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, return_type="dicts")
        check_flag = all(isinstance(item, dict) for item in df_transformed)
        expected = True
        actual = check_flag

        assert expected == actual

    def test_expected_output_single_dict(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "single_dict" """
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, return_type="single_dict")
        check_flag = isinstance(df_transformed, dict)
        expected = True
        actual = check_flag

        assert expected == actual

    # sorting types
    def test_expected_output_abs(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when sorting = "abs" """
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, sorting="abs", return_type="dicts")
        newlist = sorted(
            df_transformed,
            key=lambda df_transformed: list(map(abs, df_transformed.values())),
            reverse=True,
        )
        expected = df_transformed
        actual = newlist

        assert expected == actual

    def test_expected_output_label_1(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when sorting = "label" and
        pred_label = 1
        """
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(
            single_row, sorting="label", pred_label=1, return_type="dicts"
        )
        newlist = sorted(
            df_transformed,
            key=lambda df_transformed: list(df_transformed.values()),
            reverse=True,
        )
        expected = df_transformed
        actual = newlist

        assert expected == actual

    def test_expected_output_label_0(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when sorting = "label" and
        pred_label = 0
        """
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(
            single_row, sorting="label", pred_label=0, return_type="dicts"
        )
        newlist = sorted(
            df_transformed,
            key=lambda df_transformed: list(df_transformed.values()),
            reverse=False,
        )
        expected = df_transformed
        actual = newlist

        assert expected == actual

    # n_return
    @given(s=st.integers().filter(lambda x: x > 0))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_expected_output_n_return(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        s: int,
        features: list,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when n_return = any integer>0"""
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(single_row, n_return=s, return_type="dicts")

        if 0 < s <= len(features):
            expected = len(df_transformed)
            actual = s
        else:
            expected = len(df_transformed)
            actual = len(features)

        assert expected == actual

    @given(s=st.booleans())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_expected_output_return_feature_values(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        s: int,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_feature_values = True or False"""
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(
            single_row, return_feature_values=s, return_type="dicts"
        )

        if not s:
            count = len(df_transformed[1].values())
            expected = count
            actual = 1
        else:
            count = len(list(df_transformed[1].values())[0])
            expected = count
            actual = 2

        assert expected == actual

    # feature_mappings
    def test_expected_output_return_feature_mappings(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when feature_mappings = mapping_dict"""
        mapping_dict = {
            "MedInc": "MedInc feature",
            "HouseAge": "Age of house",
            "AveRooms": "Average number of rooms",
            "AveBedrms": "Average number of bedrooms",
            "Population": "Population feature",
            "AveOccup": "Average occupation",
        }
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(
            single_row, feature_mappings=mapping_dict, return_type="dicts"
        )

        all_keys = set().union(
            *(df_transformed.keys() for df_transformed in df_transformed)
        )

        check_flag = set(list(all_keys)) == set(list(mapping_dict.values()))
        expected = True
        actual = check_flag
        assert expected == actual

    def test_expected_output_return_group_mappings(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when feature_mappings = grouping_dict"""
        grouping_dict = {
            "MedInc": "MedInc feature",
            "HouseAge": "Age of house",
            "AveRooms": "Number of rooms",
            "AveBedrms": "Number of rooms",
            "Population": "Population feature",
            "AveOccup": "Average occupation",
        }
        MI.fit(lin_model, lin_X_train)
        df_transformed = MI.transform(
            single_row, feature_mappings=grouping_dict, return_type="dicts"
        )

        all_keys = set().union(
            *(df_transformed.keys() for df_transformed in df_transformed)
        )
        check_flag = set(list(all_keys)) == set(list(grouping_dict.values()))
        expected = True
        actual = check_flag

        assert expected == actual

    def test_expected_output_one_hot_encoding(
        self,
        tree_model_one_hot,
        one_hot_tree_X_test: pd.Series,
        output_one_hot_encoding: list,
        one_hot_features: list,
    ):
        """Test that transform returns the expected output when one hot columns are passed"""
        x = ModelInterpreter(feature_names=one_hot_features, one_hot_cols=["Over25"])
        x.fit(tree_model_one_hot)
        single_row = one_hot_tree_X_test[one_hot_features].head(1)
        df_transformed = x.transform(
            single_row, return_feature_values=True, return_type="dicts"
        )

        expected = df_transformed
        actual = output_one_hot_encoding

        assert expected == actual

    def test_error_sorting_input_not_supported(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check ValueError if x.transform() sorting method is not supported"""
        with pytest.raises(
            ValueError,
            match="sorting method not supported",
        ):
            MI.fit(lin_model, lin_X_train)
            MI.transform(single_row, sorting="XYZ", return_type="dicts")

    def test_error_return_type_not_supported(
        self,
        MI: ModelInterpreter,
        lin_model,
        lin_X_train: pd.Series,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check ValueError if x.transform() return_type method is not supported"""
        with pytest.raises(
            ValueError,
            match="return_type not supported",
        ):
            MI.fit(lin_model, lin_X_train)
            MI.transform(single_row, return_type="XYZ")
