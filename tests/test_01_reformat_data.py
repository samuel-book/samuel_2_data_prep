import os
import pandas as pd
import unittest

# Run from command line: python test_01_reformat_data.py


class DataTests(unittest.TestCase):
    '''Class for running unit tests'''

    # Run set up once for whole class
    @classmethod
    def setUpClass(self):
        '''Set up - runs prior to each test'''
        # Paths and filenames
        raw_path: str = './../data'
        raw_filename: str = 'SAMueL ssnap extract v2.csv'
        clean_path: str = './../output'
        clean_filename: str = 'reformatted_data.csv'
        # Import dataframes
        raw_data = pd.read_csv(os.path.join(raw_path, raw_filename),
                               low_memory=False)
        clean_data = pd.read_csv(os.path.join(clean_path, clean_filename),
                                 low_memory=False)
        # Save to DataTests class
        self.raw = raw_data
        self.clean = clean_data

    def freq(self, raw_col, raw_val, clean_col, clean_val):
        '''
        Test that the frequency of a value in the raw data is same as
        the frequency of a value in the cleaned data
        Inputs:
        - self
        - raw_col and clean_col = string
        - raw_val and clean_val = string, number, or list
        Performs assertEqual test.
        '''
        # If values are not lists, convert to lists
        if type(raw_val) != list:
            raw_val = [raw_val]
        if type(clean_val) != list:
            clean_val = [clean_val]
        # Find frequencies and check if equal
        raw_freq = (self.raw[raw_col].isin(raw_val).values).sum()
        clean_freq = (self.clean[clean_col].isin(clean_val).values).sum()
        self.assertEqual(raw_freq, clean_freq)

    def test_raw_shape(self):
        '''Test the raw dataframe shape is as expected'''
        self.assertEqual(self.raw.shape, (360381, 83))

    def test_gender(self):
        '''Test the number of each gender has not changed'''
        self.freq('S1Gender', 'M', 'male', 1)
        self.freq('S1Gender', 'F', 'male', 0)

    def test_stroke_type(self):
        '''Test the number of each gender has not changed'''
        self.freq('S2StrokeType', 'I', 'infarction', 1)
        self.freq('S2StrokeType', 'PIH', 'infarction', 0)

    def test_onset(self):
        '''Test that onset_known is equal to precise + best estimate'''
        self.freq('S1OnsetTimeType', ['P', 'BE'], 'onset_known', 1)
        self.freq('S1OnsetTimeType', 'NK', 'onset_known', 0)

    def test_precise_onset(self):
        '''Test that precise_onset_known is 1 for P and 0 for BE + NK'''
        self.freq('S1OnsetTimeType', 'P', 'precise_onset_known', 1)
        self.freq('S1OnsetTimeType', ['BE', 'NK'], 'precise_onset_known', 0)

    def test_sleep(self):
        '''Test that onset during sleep values are consistent'''
        self.freq('S1OnsetDateType', 'DS', 'onset_during_sleep', 1)
        self.freq('S1OnsetDateType', ['P', 'BE'], 'onset_during_sleep', 0)

    def test_by_ambulance(self):
        '''Test if numbers arriving by ambulance are correct'''
        self.freq('S1ArriveByAmbulance', 'Y', 'arrive_by_ambulance', 1)
        self.freq('S1ArriveByAmbulance', 'N', 'arrive_by_ambulance', 0)


if __name__ == '__main__':
    unittest.main()
