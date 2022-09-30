import numpy as np
import xgboost as xgb
from sklearn import metrics

from pyutils.analysis.common.data_transformers import Transformers
from pyutils.common.strings import S_PREDICTED_RUNTIME


class MetricPrettyName:
    rel_err_per = 'Relative Error %'
    abs_rel_err_per = 'Abs. Relative Error %'
    rel_difference_per = 'Relative Difference %'

    @staticmethod
    def get_pretty_evaluation_metrics(input_results):
        pretty_results = dict()
        pretty_results['5%'] = float(input_results['percentage-with-error-less-than-5%'])
        pretty_results['10%'] = float(input_results['percentage-with-error-less-than-10%'])
        pretty_results['15%'] = float(input_results['percentage-with-error-less-than-15%'])
        pretty_results['20%'] = float(input_results['percentage-with-error-less-than-20%'])
        pretty_results['RMSE'] = float(input_results['real-rmse'])
        pretty_results['MAE'] = float(input_results['real-mae'])
        return pretty_results


class Metrics:
    @staticmethod
    def relative_error_percentage(y_true, y_pred, absolute=True):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        percentage_error = 100 * (y_pred - y_true) / y_true
        if absolute:
            return np.abs(percentage_error)
        return percentage_error

    @staticmethod
    def max_abs_error(y_true, y_pred):
        return metrics.max_error(y_true, y_pred)

    @staticmethod
    def percentage_of_error_less_than_threshold(y_true, y_pred, threshold=5, log=True):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        percentage_error = np.abs((y_true - y_pred) / y_true) * 100
        indexes = np.where(percentage_error <= threshold)
        count = len(percentage_error[indexes])
        percentage_count = float(100 * count / len(y_pred))
        if log:
            print(f"Ratio of predictions with error less than {threshold}% is: {count}/{len(y_pred)} -> "
                  f"{int(percentage_count)}%")
        return threshold, percentage_count

    @staticmethod
    def percentage_less_than_5(estimator, x_test, y_true):
        y_pred = estimator.predict(x_test)

        y_pred = Transformers.inverse_log_transform(y_pred)
        y_true = Transformers.inverse_log_transform(y_true)

        _, count = Metrics.percentage_of_error_less_than_threshold(y_true, y_pred, threshold=5, log=False)
        return count

    @staticmethod
    def mean_abs_error_percentage(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def max_abs_error_percentage(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.max(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def abs_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.abs(y_true - y_pred)

    @staticmethod
    def calculate_and_report_error_stats(data, label):
        result = {'rmse': metrics.mean_squared_error(data[label], data[S_PREDICTED_RUNTIME], squared=False),
                  'mae': metrics.mean_absolute_error(data[label], data[S_PREDICTED_RUNTIME]),
                  'max': data['Error %'].max(), 'min': data['Error %'].min(), 'mean': data['Error %'].mean(),
                  'std': data['Error %'].std()}

        print("RMSE:            ", "{:.2f}".format(result['rmse']))
        print("MAE:             ", "{:.2f}".format(result['mae']))
        print("Max Error %:     ", "{:.2f}".format(result['max']))
        print("Min Error %:     ", "{:.2f}".format(result['min']))
        print("Mean Error %:    ", "{:.2f}".format(result['mean']))
        print("STD of Error %:  ", "{:.2f}".format(result['std']))

        for i in [5, 10, 15, 20]:
            y_true = Transformers.inverse_log_transform(data[label])
            y_pred = Transformers.inverse_log_transform(data[S_PREDICTED_RUNTIME])
            _, result[f'threshold_{i}'] = \
                Metrics.percentage_of_error_less_than_threshold(y_true, y_pred, threshold=i)
        return result

    @staticmethod
    def rmse(y_true, y_pred):
        return metrics.mean_squared_error(y_true, y_pred, squared=False)

    @staticmethod
    def rmspe(y_true, y_pred):
        return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0)) * 100

    @staticmethod
    def mae(y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def combined_prediction_evaluation_metrics(y_true, y_pred, as_type='list'):
        _, count_5 = Metrics.percentage_of_error_less_than_threshold(y_true, y_pred, threshold=5, log=False)
        _, count_10 = Metrics.percentage_of_error_less_than_threshold(y_true, y_pred, threshold=10, log=False)
        _, count_15 = Metrics.percentage_of_error_less_than_threshold(y_true, y_pred, threshold=15, log=False)
        _, count_20 = Metrics.percentage_of_error_less_than_threshold(y_true, y_pred, threshold=20, log=False)

        # This is to report the error stats on the real data without the effect of the transformation
        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)

        if as_type == 'list':
            return [
                ('real-rmse', rmse),
                ('real-mae', mae),
                ('percentage-with-error-less-than-10%', count_10),
                ('percentage-with-error-less-than-15%', count_15),
                ('percentage-with-error-less-than-20%', count_20),
                ('percentage-with-error-less-than-5%', count_5)

            ]
        elif as_type == 'pretty_dict':
            return {'5%': count_5, '10%': count_10, '15%': count_15, '20%': count_20,
                    'RMSE': rmse, 'MAE': mae}
        else:
            return None

    class XGBoostEval:
        @staticmethod
        def percentage_error_from_dmatrix(y_pred, data: xgb.DMatrix):
            y_true = data.get_label()

            y_true = Transformers.inverse_log_transform(y_true)
            y_pred = Transformers.inverse_log_transform(y_pred)

            return Metrics.combined_prediction_evaluation_metrics(y_true, y_pred)

        @staticmethod
        def rmse_from_dmatrix(estimator, x_test, y_true):
            y_pred = estimator.predict(x_test)

            y_pred = Transformers.inverse_log_transform(y_pred)
            y_true = Transformers.inverse_log_transform(y_true)

            return -1 * Metrics.rmse(y_true, y_pred)
