import unittest
from DecisionTree_ver4 import addResionLabel
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
        
if __name__ == "__main__":
    unittest.main()
        
