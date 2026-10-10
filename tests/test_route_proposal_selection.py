import unittest
from BalloonPoppingGymEnv.agents.route_proposal_selection import RouteProposal, choose_route


class ProposalSelectionTests(unittest.TestCase):
    def test_shorter_route_can_win_on_predicted_actual_hits(self):
        long=RouteProposal((1,2,3,4),(2.,2.,2.,2.),1.2)
        short=RouteProposal((5,6),(2.,2.),1.8)
        self.assertIs(choose_route([long,short],10.,lambda _:True),short)

    def test_high_value_never_bypasses_physical_validation(self):
        impossible=RouteProposal((1,2),(2.,2.),2.)
        feasible=RouteProposal((3,),(3.,),.9)
        self.assertIs(choose_route([impossible,feasible],10.,lambda p:p is feasible),feasible)
        self.assertIsNone(choose_route([impossible],10.,lambda _:False))

    def test_invalid_predictions_timings_and_duplicate_targets_are_rejected(self):
        proposals=[RouteProposal((1,1),(1.,1.),2.),RouteProposal((1,),(20.,),1.),
                   RouteProposal((1,),(-1.,),1.),RouteProposal((1,),(1.,),float('nan'))]
        called=[]
        self.assertIsNone(choose_route(proposals,5.,lambda p:called.append(p)))
        self.assertEqual(called,[])


if __name__ == '__main__':
    unittest.main()
