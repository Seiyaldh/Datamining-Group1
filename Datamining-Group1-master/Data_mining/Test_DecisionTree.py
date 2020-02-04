import unittest
from DecisionTree import addResionLabel
from DecisionTree import quartile
from pandas.util.testing import assert_frame_equal
import pandas as pd
import numpy as np

class TestDecisionTree(unittest.TestCase):
    def test_addResionLabel(self):
        label = pd.DataFrame({  'Country' : ['USA'],
                                'Region' : ['North America'],
                                'region label' : np.array([4])})
                                
        df = pd.DataFrame({ 'Country' : ['USA'],
                            'Region' : ['North America']})
                            
        df = addResionLabel(df)
        assert_frame_equal(label, df)

        def test_quartile(self):
            df = pd.DataFrame({"Happiness Score": np.array([0, 1, 2, 3])})
            exp = df
            exp["label"] = pd.DataFrame([0, 1, 2, 3])
            act = quartile(df)
            assert_frame_equal(exp, act)
        
if __name__ == "__main__":
    unittest.main()
        
