import pandas as pd
import sys

# FALTA AÑADIR LOS PARAMETROIS DE DESCRIBE1º

class   Describe():
    def __init__(self, data: pd.DataFrame):
        self.pq1 = 0.25
        self.pq2 = 0.5
        self.pq3 = 0.75
        self.df = data
        self.get_stats()

    def get_min_max(self, clean_df):
        min = max = clean_df.iloc[0]
        for item in clean_df:
            if item < min:
                min = item
            if item > max:
                max = item
        return min, max
    
    def percentile(self, data, percent):
        data_sorted = sorted(data)
        if len(data_sorted) == 0:
            return float('nan')
        k = (len(data_sorted) - 1) * percent
        f = int(k)
        c = f + 1
        if c >= len(data_sorted):
            return data_sorted[f]
        return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


    def get_stats(self):

        # Gets features with numeric values
        df_to_stats = self.df.select_dtypes(include=['float', 'int'])

        count, mean, std, min, max, pq1, pq2, pq3, index = [], [], [], [], [], [], [], [], []


        for feature in df_to_stats.columns:
            index.append(feature)
            # remove nan values
            clean_df = df_to_stats[feature].dropna().astype(float)

            # if there are no valid values
            if clean_df.__len__() == 0:
                count.append(0)
                mean.append(float('nan'))
                std.append(float('nan'))
                min.append(float('nan'))
                max.append(float('nan'))
                pq1.append(float('nan'))
                pq2.append(float('nan'))
                pq3.append(float('nan'))
                continue
            
            # Count
            count.append(clean_df.__len__())

            # Mean
            mean.append(clean_df.sum() / clean_df.__len__())

            # Std
            r = []
            for item in clean_df:
                r.append((item - (clean_df.sum() / clean_df.__len__())) ** 2)
            s = (sum(r) / (clean_df.__len__() - 1)) ** 0.5
            std.append(s)

            # Min, max
            number_min, number_max = self.get_min_max(clean_df)
            min.append(number_min)
            max.append(number_max)

            # Percentiles
            pq1.append(self.percentile(clean_df, self.pq1))
            pq2.append(self.percentile(clean_df, self.pq2))
            pq3.append(self.percentile(clean_df, self.pq3))

        stats_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        df_count = pd.DataFrame(count, index=index, columns=[stats_names[0]]).transpose()
        df_mean = pd.DataFrame(mean, index=index, columns=[stats_names[1]]).transpose()
        df_std = pd.DataFrame(std, index=index, columns=[stats_names[2]]).transpose()
        df_min = pd.DataFrame(min, index=index, columns=[stats_names[3]]).transpose()
        df_pq1 = pd.DataFrame(pq1, index=index, columns=[stats_names[4]]).transpose()
        df_pq2 = pd.DataFrame(pq2, index=index, columns=[stats_names[5]]).transpose()
        df_pq3 = pd.DataFrame(pq3, index=index, columns=[stats_names[6]]).transpose()
        df_max = pd.DataFrame(max, index=index, columns=[stats_names[7]]).transpose()

        self.result = pd.concat([df_count, df_mean, df_std, df_min, df_pq1, df_pq2, df_pq3, df_max])
        # print(self.result)

    def print(self):
        print(self.result)