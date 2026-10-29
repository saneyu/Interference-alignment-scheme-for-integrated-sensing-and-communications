import unittest
import numpy as np
from isac.shared_source import make_case,shared_power,sensing
from isac.qpsk_receiver import posterior,score,communication
from isac.models import qpsk,cn

FAMILIES=['dft','lfm','walsh','qpsk']


class SharedSourceTests(unittest.TestCase):
    def test_identical_symbol_domains_and_energies(self):
        power=shared_power(FAMILIES,[.3,.5,.7,.9])
        for scenario,nmsg,slots in [('BIA',6,6),('TIM',20,10)]:
            for family in FAMILIES:
                c=make_case(scenario,family,.7,power)
                self.assertEqual(len(c.owners),nmsg)
                self.assertEqual(c.encoded.communication.shape[2],nmsg)
                self.assertEqual(c.direct.communication.shape[2],nmsg)
                self.assertEqual(c.encoded.slots,slots)
                self.assertEqual(c.direct.slots,4)
                np.testing.assert_allclose(np.sum(abs(c.encoded.communication)**2,axis=(0,1)),
                                           np.sum(abs(c.direct.communication)**2,axis=(0,1)),rtol=1e-12)
                np.testing.assert_allclose(np.sum(abs(c.encoded.sensing)**2,axis=0),
                                           np.sum(abs(c.direct.sensing)**2,axis=0),rtol=1e-12,atol=1e-13)
                for f in (c.encoded,c.direct):
                    self.assertLessEqual(np.max(abs(f.sensing)**2+np.sum(abs(f.communication)**2,axis=2)),1+1e-12)

    def test_literal_direct_superposition(self):
        for scenario in ('BIA','TIM'):
            c=make_case(scenario,'lfm',.7,.2)
            scaled=c.direct.sensing[:,c.sensed]/c.raw_sensing
            np.testing.assert_allclose(scaled,np.broadcast_to(scaled[:1],scaled.shape),atol=1e-12)
            for j in range(len(c.owners)):
                nz=np.flatnonzero(np.sum(abs(c.direct.communication[:,:,j])**2,axis=1)>0)
                self.assertEqual(len(nz),2 if scenario=='BIA' else 1)

    def test_qpsk_posterior_and_rate_ceiling(self):
        rng=np.random.default_rng(5);s,b=qpsk(rng,(300,1))
        a=np.ones((300,1,1),complex);sigma2=1e-4
        y=s+np.sqrt(sigma2)*cn(rng,s.shape)
        lp,labels=posterior(y,a,[0],sigma2)
        info,errors=score(lp,labels,b)
        self.assertEqual(errors.sum(),0)
        np.testing.assert_allclose(info,2,atol=1e-10)
        np.testing.assert_allclose(np.exp(lp).sum(axis=1),1,atol=1e-12)
        lp2,_=posterior(y*np.sqrt(3),a*np.sqrt(3),[0],sigma2*3)
        np.testing.assert_allclose(lp,lp2,rtol=1e-11,atol=1e-8)

    def test_joint_receiver_noiseless_limit_and_normalization(self):
        for scenario,nmsg in [('BIA',6),('TIM',20)]:
            c=make_case(scenario,'dft',.7,.2)
            out=communication(c,80,30,91,batch=15)
            for name in ('encoded','direct'):
                self.assertEqual(out[name]['bit_errors'].sum(),0)
                np.testing.assert_allclose(out[name]['information_bits_per_frame'],2*nmsg,atol=1e-7)
                np.testing.assert_allclose(out[name]['rate_common_window'],2*nmsg/c.encoded.slots,atol=1e-7)
                self.assertLess(out[name]['cancellation_error'],1e-12)
            self.assertTrue(all(r['zf_feasible'] for r in out['rank_audit'] if r['format']=='encoded'))
            self.assertTrue(all(not r['zf_feasible'] for r in out['rank_audit'] if r['format']=='direct'))

    def test_sensing_theory_with_repeated_baseline_data(self):
        for scenario in ('BIA','TIM'):
            c=make_case(scenario,'dft',.7,.2)
            errors,theory,_=sensing(c,20,12000,31)
            for name in (scenario,'TIN'):
                self.assertAlmostEqual(errors[name].mean()/theory[name],1,delta=.05)


if __name__=='__main__':unittest.main()
