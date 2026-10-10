import unittest
import torch
from BalloonPoppingGymEnv.agents.route_attention_model import RouteAttentionScorer


class RouteAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model=RouteAttentionScorer(width=32,heads=4,layers=2).eval()

    def test_permutation_equivariance_and_masks(self):
        nodes=torch.randn(1,5,12)
        context=torch.randn(1,9)
        visible=torch.tensor([[True,True,True,False,False]])
        candidates=torch.tensor([[True,False,True,False,False]])
        permutation=torch.tensor([2,4,0,1,3])
        with torch.no_grad():
            a,v=self.model(nodes,context,visible,candidates)
            b,w=self.model(nodes[:,permutation],context,visible[:,permutation],candidates[:,permutation])
        torch.testing.assert_close(a[:,:5][:,permutation],b[:,:5],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(a[:,-1],b[:,-1],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(v,w,rtol=1e-5,atol=1e-5)
        self.assertEqual(torch.softmax(a,dim=-1)[0,1],0)
        self.assertTrue(torch.isfinite(v).all())

    def test_hidden_padding_and_no_target_are_safe(self):
        nodes=torch.full((1,3,12),float('nan'))
        mask=torch.zeros(1,3,dtype=torch.bool)
        logits,value=self.model(nodes,torch.zeros(1,9),mask,mask)
        torch.testing.assert_close(torch.softmax(logits,dim=-1),torch.tensor([[0.,0.,0.,1.]]))
        self.assertTrue(torch.isfinite(value).all())
        logits,value=self.model(nodes[:,:0],torch.zeros(1,9),mask[:,:0],mask[:,:0])
        self.assertEqual(logits.shape,(1,1))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gradients_and_visible_nan_rejection(self):
        mask=torch.ones(2,3,dtype=torch.bool)
        nodes=torch.randn(2,3,12)
        logits,value=self.model(nodes,torch.zeros(2,9),mask,mask)
        loss=-torch.log_softmax(logits,dim=-1)[:,0].mean()+value.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        nodes[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            self.model(nodes,torch.zeros(2,9),mask,mask)


if __name__ == '__main__':
    unittest.main()
