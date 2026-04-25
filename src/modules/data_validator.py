import pandas as pd
import numpy as np
from datetime import datetime


class DataValidator:

    def __init__(self):
        self.validation_results = []

    def validate_schema(self, df, expected_columns, expected_types):
        """Check column presence and data types."""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Schema',
            'passed': True,
            'issues': [],
        }

        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            report['passed'] = False
            report['issues'].append(f"Missing columns: {missing_cols}")

        # Check for unexpected extra columns
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            report['issues'].append(f"Extra columns (not in schema): {extra_cols}")

        # Check data types match expected
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    report['passed'] = False
                    report['issues'].append(
                        f"Column '{col}': expected type '{expected_type}', got '{actual_type}'"
                    )

        self.validation_results.append(report)
        return report

    def validate_completeness(self, df, required_columns, max_missing_pct=0.05):
        """Check that required columns don't have too many missing values."""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Completeness',
            'passed': True,
            'issues': [],
        }

        for col in required_columns:
            if col not in df.columns:
                report['passed'] = False
                report['issues'].append(f"Required column '{col}' not found in data")
                continue

            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            if missing_pct > max_missing_pct * 100:
                report['passed'] = False
                report['issues'].append(
                    f"Column '{col}': {missing_pct:.2f}% missing "
                    f"(allowed max: {max_missing_pct * 100:.0f}%)"
                )

        self.validation_results.append(report)
        return report

    def validate_ranges(self, df, range_rules):
        """Check that numeric values fall within expected min/max bounds."""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Ranges',
            'passed': True,
            'issues': [],
        }

        for col, rules in range_rules.items():
            if col not in df.columns:
                continue

            if 'min' in rules:
                below_min = df[df[col] < rules['min']]
                if len(below_min) > 0:
                    report['passed'] = False
                    report['issues'].append(
                        f"Column '{col}': {len(below_min)} values below minimum ({rules['min']})"
                    )

            if 'max' in rules:
                above_max = df[df[col] > rules['max']]
                if len(above_max) > 0:
                    report['passed'] = False
                    report['issues'].append(
                        f"Column '{col}': {len(above_max)} values above maximum ({rules['max']})"
                    )

        self.validation_results.append(report)
        return report

    def validate_uniqueness(self, df, unique_columns):
        """Check that key columns have no duplicate values."""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Uniqueness',
            'passed': True,
            'issues': [],
        }

        for col in unique_columns:
            if col not in df.columns:
                continue

            duplicate_count = df[col].duplicated().sum()
            if duplicate_count > 0:
                report['passed'] = False
                report['issues'].append(
                    f"Column '{col}': {duplicate_count} duplicate values found"
                )

        self.validation_results.append(report)
        return report

    def validate_categorical(self, df, categorical_rules):
        """Check that categorical columns only contain allowed values."""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Categorical',
            'passed': True,
            'issues': [],
        }

        for col, allowed_values in categorical_rules.items():
            if col not in df.columns:
                continue

            actual_values = set(df[col].dropna().unique())
            invalid_values = actual_values - set(allowed_values)

            if invalid_values:
                report['passed'] = False
                report['issues'].append(
                    f"Column '{col}': invalid values found: {invalid_values}. "
                    f"Allowed: {set(allowed_values)}"
                )

        self.validation_results.append(report)
        return report

    def validate_outliers_iqr(self, df, columns):
        """
        Detect outliers in numeric columns using the IQR method.
        Flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        Mirrors the IQR outlier checks done in the notebook for price and kilometer.
        """
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Outliers (IQR)',
            'passed': True,
            'issues': [],
        }

        for col in columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            outlier_pct = (outlier_count / len(df)) * 100

            if outlier_count > 0:
                report['passed'] = False
                report['issues'].append(
                    f"Column '{col}': {outlier_count} outliers detected "
                    f"({outlier_pct:.2f}%) — bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

        self.validation_results.append(report)
        return report

    def validate_duplicates(self, df):
        """
        Check for fully duplicated rows across all columns.
        Mirrors the duplicate-row check done in the notebook.
        """
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'check_type': 'Duplicate Rows',
            'passed': True,
            'issues': [],
        }

        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100

        if duplicate_count > 0:
            report['passed'] = False
            report['issues'].append(
                f"{duplicate_count} fully duplicate rows found ({duplicate_pct:.2f}%)"
            )

        self.validation_results.append(report)
        return report

    def generate_report(self):
        """Print a clear, readable validation summary report."""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r['passed'])
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0

        print("=" * 55)
        print("       DATA VALIDATION REPORT")
        print("=" * 55)
        print(f"  Total Checks  : {total}")
        print(f"  Passed        : {passed}")
        print(f"  Failed        : {failed}")
        print(f"  Success Rate  : {success_rate:.1f}%")
        print("=" * 55)

        for result in self.validation_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"\n[{status}] {result['check_type']} Check")
            print(f"   Time: {result['timestamp']}")

            if result['issues']:
                for issue in result['issues']:
                    print(f"   ⚠  {issue}")
            else:
                print("   ✓  No issues found.")

        print("\n" + "=" * 55)

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': success_rate,
            'details': self.validation_results,
        }


raw_data  = pd.read_csv("../../data/joined_cars_dataset.csv")

expected_columns = [
    'price', 'yearOfRegistration', 'powerPS',
    'kilometer', 'price_tier', 'fuelType', 
    'vehicleType', 'brand', 'gearbox', 'seller'
]

expected_types = {
    'price':               'int64',
    'yearOfRegistration':  'float64',
    'powerPS':             'float64',
    'kilometer':           'float64',
    'price_tier':          'object',
    'fuelType':           'object',
    'vehicleType':        'object',
    'brand':              'object',
    'gearbox':           'object',
    'seller':            'object',
}

required_columns = ['price', 'yearOfRegistration', 'powerPS', 'kilometer']

range_rules = {
    'price':               {'min': 0},
    'powerPS':             {'min': 1, 'max': 5000},
    'kilometer':           {'min': 0, 'max': 200000},
    'yearOfRegistration':  {'min': 1900, 'max': datetime.now().year},
}

categorical_rules = {
    'price_tier': ['budget', 'mid-range', 'premium']
}

outlier_columns = ['price', 'kilometer']
validator = DataValidator()

validator.validate_schema(raw_data, expected_columns, expected_types)
validator.validate_completeness(raw_data, required_columns, max_missing_pct=0.05)
validator.validate_ranges(raw_data, range_rules)
validator.validate_categorical(raw_data, categorical_rules)
validator.validate_outliers_iqr(raw_data, outlier_columns)
validator.validate_duplicates(raw_data)

validator.generate_report()