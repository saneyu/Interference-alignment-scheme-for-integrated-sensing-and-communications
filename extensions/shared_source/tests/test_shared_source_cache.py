import unittest
import numpy as np
from isac.shared_source import make_case
from isac.qpsk_receiver import communication


class SharedSourceCacheTests(unittest.TestCase):
    def test_fixed_sensing_family_does_not_change_communication(self):
        for scenario in ('BIA','TIM'):
            reference=communication(make_case(scenario,'dft',.7,.2),10,20,21,batch=10)
            other=communication(make_case(scenario,'lfm',.7,.2),10,20,21,batch=10)
            for name in ('encoded','direct'):
                np.testing.assert_array_equal(reference[name]['bit_errors'],other[name]['bit_errors'])
                np.testing.assert_allclose(reference[name]['information_bits_per_frame'],
                                           other[name]['information_bits_per_frame'],rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main()
