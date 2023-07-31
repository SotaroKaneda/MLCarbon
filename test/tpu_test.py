import unittest
import pandas as pd

class Tpu_test(unittest.TestCase):
    def test(self):
        one = 1
        two = 2
        self.assertEqual(two - one, 1, "incorrect math")

unittest.main()
models_df = pd.read_csv('./data/models.csv')
tokens_t = [0.0099, 0.0099,4.009984,1,10,1.2,0.013924]
models_df['Tokens(trillions)'] = tokens_t
models_df.to_csv('models.csv', sep=',', index=False)