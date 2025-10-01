import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from week5_706 import (
    check_duplicates,
    decimal_format,
    group_by_regions,
    train_tree_model,
    update_country_regions,
    year_range,
)


class TestDecimalFormat(unittest.TestCase):
    def test_integer_and_float_strings(self):
        df = pd.DataFrame({"col": ["1234", "567,89"]})
        result = decimal_format(df.copy(), "col")
        self.assertTrue(np.issubdtype(result["col"].dtype, np.floating))
        self.assertAlmostEqual(result["col"].iloc[0], 1234.0)
        self.assertAlmostEqual(result["col"].iloc[1], 567.89)

    def test_already_float(self):
        df = pd.DataFrame({"col": [1.23, 4.56]})
        result = decimal_format(df.copy(), "col")
        self.assertTrue(np.allclose(result["col"], [1.23, 4.56]))


class TestUpdateCountryRegions(unittest.TestCase):
    def test_updates_to_latest_region(self):
        df = pd.DataFrame(
            {
                "Country": ["A", "A", "B"],
                "Regional indicator": ["X", "Y", "Z"],
                "Year": [2000, 2005, 2001],
            }
        )
        updated = update_country_regions(df.copy())
        # "A" should have region "Y" (latest year).
        self.assertTrue(
            (updated.loc[updated["Country"] == "A", "Regional indicator"] == "Y").all()
        )

    def test_no_change_when_consistent(self):
        df = pd.DataFrame(
            {
                "Country": ["C", "C"],
                "Regional indicator": ["M", "M"],
                "Year": [2000, 2001],
            }
        )
        updated = update_country_regions(df.copy())
        self.assertTrue((updated["Regional indicator"] == "M").all())


class TestCheckDuplicates(unittest.TestCase):
    def test_no_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        self.assertEqual(check_duplicates(df), 0)

    def test_with_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]})
        self.assertEqual(check_duplicates(df), 3)


class TestYearRange(unittest.TestCase):
    def test_valid_range(self):
        df = pd.DataFrame({"Year": [2000, 2001, 2002, 2003], "val": [1, 2, 3, 4]})
        filtered = year_range(df, range(2001, 2003))
        self.assertEqual(set(filtered.keys()), {2001, 2002})
        self.assertTrue((filtered[2001]["Year"] == 2001).all())

    def test_empty_for_missing_years(self):
        df = pd.DataFrame({"Year": [2000, 2001], "val": [5, 6]})
        filtered = year_range(df, [2010])
        self.assertIn(2010, filtered)
        self.assertEqual(len(filtered[2010]), 0)


class TestGroupByRegions(unittest.TestCase):
    def test_grouping(self):
        df = pd.DataFrame(
            {"Regional indicator": ["Asia", "Asia", "Europe"], "Value": [10, 20, 30]}
        )
        grouped = group_by_regions(df)
        self.assertIn("Asia", grouped)
        self.assertEqual(grouped["Asia"]["Value"].sum(), 30)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Regional indicator", "Value"])
        grouped = group_by_regions(df)
        self.assertEqual(grouped, {})


class TestTrainTreeModel(unittest.TestCase):
    def test_model_training_and_prediction(self):
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        model, X_test, y_test = train_tree_model(df, "target")
        self.assertIsInstance(model, DecisionTreeRegressor)
        preds = model.predict(X_test)
        self.assertEqual(len(preds), len(y_test))

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["feature1", "target"])
        with self.assertRaises(ValueError):
            train_tree_model(df, "target")

    def test_raises_on_missing_target(self):
        df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        with self.assertRaises(ValueError):
            train_tree_model(df, "target")


class TestSystemPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        df = pd.DataFrame(
            {
                "Country": ["A", "A", "B"],
                "Regional indicator": ["X", "Y", "Z"],
                "Year": [2000, 2005, 2001],
                "Value": ["1,234", "2,345", "3,456"],
                "target": [10, 20, 30],
            }
        )

        df = decimal_format(df, "Value")

        df = update_country_regions(df)

        self.assertEqual(check_duplicates(df), 0)

        grouped = group_by_regions(df)
        self.assertIn("Y", grouped.keys())
        self.assertIn("Z", grouped.keys())

        model, X_test, y_test = train_tree_model(df[["Value", "target"]], "target")
        preds = model.predict(X_test)

        self.assertEqual(len(preds), len(y_test))


unittest.main(argv=[""], exit=False)
