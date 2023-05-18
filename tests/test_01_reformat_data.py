import os
import pandas as pd
import unittest

# This is currently run by notebook 01_reformat_data.ipynb


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

    def time_neg(self, time_column):
        '''
        Function for testing that times are not negative when expected
        to be positive.
        Input: time_column = string, column with times
        '''
        self.assertEqual(sum(self.clean[time_column] < 0), 0)

    def equal_array(self, df, col, exp_array):
        '''
        Function to check that the only possible values in a column are
        those provided by exp_array.
        Inputs:
        - df = dataframe (raw or clean)
        - col = string (column name)
        - exp_array = array (expected values for column)
        '''
        # Sorted so that array order does not matter
        self.assertEqual(sorted(df[col].unique()), sorted(exp_array))

    def test_raw_shape(self):
        '''Test the raw dataframe shape is as expected'''
        self.assertEqual(self.raw.shape, (360381, 83))

    def test_id(self):
        '''Test that ID numbers are all unique'''
        self.assertEqual(len(self.clean.id.unique()),
                         len(self.clean.index))

    def test_time_negative(self):
        '''Test that times are not negative when expected to be positive'''
        time_col = ['onset_to_arrival_time',
                    'call_to_ambulance_arrival_time',
                    'ambulance_on_scene_time',
                    'ambulance_travel_to_hospital_time',
                    'ambulance_wait_time_at_hospital',
                    'scan_to_thrombolysis_time',
                    'arrival_to_thrombectomy_time']
        for col in time_col:
            with self.subTest(msg=col):
                self.time_neg(col)

    def test_no_ambulance(self):
        '''
        Test that people who do not arrive by ambulance therefore have
        no ambulance times
        '''
        amb_neg = self.clean[(self.clean['arrive_by_ambulance'] == 0) & (
            (self.clean['call_to_ambulance_arrival_time'].notnull()) |
            (self.clean['ambulance_on_scene_time'].notnull()) |
            (self.clean['ambulance_travel_to_hospital_time'].notnull()) |
            (self.clean['ambulance_wait_time_at_hospital'].notnull()))]
        self.assertEqual(len(amb_neg.index), 0)

    def test_nihss_min_max(self):
        '''Test that minimum and maximum NIHSS scores are within bounds'''
        # NIHSS scores on arrival all between 0 and 42
        self.assertEqual(self.clean['stroke_severity'].min(), 0)
        self.assertEqual(self.clean['stroke_severity'].max(), 42)

    def test_nihss_values(self):
        '''Test that NIHSS values are as expected'''
        expected = ([
            ['nihss_arrival_loc', [0, 1, 2, 3]],
            ['nihss_arrival_loc_questions', [0, 1, 2, -1]],
            ['nihss_arrival_loc_commands', [0, 1, 2, -1]],
            ['nihss_arrival_best_gaze', [0, 1, 2, -1]],
            ['nihss_arrival_visual', [0, 1, 2, 3, -1]],
            ['nihss_arrival_facial_palsy', [0, 1, 2, 3, -1]],
            ['nihss_arrival_motor_arm_left', [0, 1, 2, 3, 4, -1]],
            ['nihss_arrival_motor_arm_right', [0, 1, 2, 3, 4, -1]],
            ['nihss_arrival_motor_leg_left', [0, 1, 2, 3, 4, -1]],
            ['nihss_arrival_motor_leg_right', [0, 1, 2, 3, 4, -1]],
            ['nihss_arrival_limb_ataxia', [0, 1, 2, -1]],
            ['nihss_arrival_sensory', [0, 1, 2, -1]],
            ['nihss_arrival_best_language', [0, 1, 2, 3, -1]],
            ['nihss_arrival_dysarthria', [0, 1, 2, -1]],
            ['nihss_arrival_extinction_inattention', [0, 1, 2, -1]]])
        for col, values in expected:
            with self.subTest(msg=col):
                self.equal_array(self.clean, col, values)

    def test_no_thrombolysis(self):
        '''
        Test that reasons for no thrombolysis only contains 0 and 1, as
        data dictionary indicates that TRUE, FALSE or empty might also be
        possible
        '''
        col_list = ['S2ThrombolysisNoButHaemorrhagic',
                    'S2ThrombolysisNoButTimeWindow',
                    'S2ThrombolysisNoButComorbidity',
                    'S2ThrombolysisNoButMedication',
                    'S2ThrombolysisNoButRefusal',
                    'S2ThrombolysisNoButAge',
                    'S2ThrombolysisNoButImproving',
                    'S2ThrombolysisNoButTooMildSevere',
                    'S2ThrombolysisNoButTimeUnknownWakeUp',
                    'S2ThrombolysisNoButOtherMedical']
        for col in col_list:
            with self.subTest(msg=col):
                self.equal_array(self.raw, col, [0, 1])


if __name__ == '__main__':
    unittest.main()
