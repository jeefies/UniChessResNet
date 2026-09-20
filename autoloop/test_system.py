import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import chess, numpy as np
from search.mcts import MCTS,MCTSConfig,Node
from search.test_mcts import uniform_evaluator
from autoloop import common
from autoloop.governor import busy

class SearchRegression(unittest.TestCase):
    def test_winning_root_sign_and_budget(self):
        b=chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1')
        search=MCTS(uniform_evaluator,MCTSConfig(simulations=400,batch_size=32))
        move,root=search.best_move(b);b.push(move)
        self.assertTrue(b.is_checkmate());self.assertGreater(search.root_value(root),0)
        self.assertEqual(int(root.N.sum()),400)
    def test_no_double_expansion_or_virtual_loss_leaks(self):
        original=Node.expand;seen=set()
        def expand(node,*args):
            self.assertFalse(node.expanded);seen.add(id(node));return original(node,*args)
        with patch.object(Node,'expand',expand):
            root=MCTS(uniform_evaluator,MCTSConfig(simulations=257,batch_size=128)).search(chess.Board())
        def check(n):
            self.assertTrue(np.all(n.VL==0))
            for c in n.children:
                if c:check(c)
        check(root);self.assertEqual(root.sum_N,257)
    def test_reused_root_is_advanced_after_real_move(self):
        calls = []
        def evaluator(boards):
            calls.append(len(boards))
            return uniform_evaluator(boards)
        board = chess.Board()
        search = MCTS(evaluator, MCTSConfig(simulations=64, batch_size=16))
        move, root = search.best_move(board)
        child = search.advance_root(root, move)
        self.assertIsNotNone(child)
        board.push(move)
        _, reused = search.best_move(board, simulations=32, root=child)
        self.assertIs(reused, child)
        self.assertTrue(search.last_metrics['reused_root'])

    def test_real_game_history_is_preserved(self):
        b=chess.Board()
        for u in ['g1f3','g8f6','f3g1','f6g8']:b.push_uci(u)
        seen=[]
        def evaluator(boards):
            seen.extend(len(x.move_stack) for x in boards);return uniform_evaluator(boards)
        MCTS(evaluator,MCTSConfig(simulations=30,batch_size=8)).search(b)
        self.assertGreater(min(seen[1:]),4)
    def test_claimed_repetition(self):
        b=chess.Board()
        for u in ['g1f3','g8f6','f3g1','f6g8']*2:b.push_uci(u)
        s=MCTS(uniform_evaluator,MCTSConfig(claim_draw=True))
        self.assertEqual(s.search(b).terminal_value,0)
    def test_cursed_tablebase_wins_are_draws(self):
        class TB:
            def probe_wdl(self,b):return 1
        b=chess.Board('8/8/8/3k4/8/8/4Q3/4K3 w - - 0 1')
        s=MCTS(uniform_evaluator,tablebase=TB());self.assertEqual(s._exact_value(b),0)
        b.halfmove_clock=99;self.assertEqual(s._exact_value(b),0)
        class WinTB:
            def probe_wdl(self,b):return 2
        s.tablebase=WinTB();self.assertIsNone(s._exact_value(b))

class OperationsRegression(unittest.TestCase):
    def test_foreign_even_low_usage_yields(self):
        self.assertTrue(busy({'processes':{1:500,2:1},'used_mib':501,'free_mib':90000},1))
    def test_own_gpu_is_not_foreign(self):
        self.assertFalse(busy({'processes':{1:6000},'used_mib':6200,'free_mib':90000},1))
    def test_unattributed_memory_yields(self):
        self.assertTrue(busy({'processes':{},'used_mib':1000,'free_mib':90000},None))
    def test_storage_and_protected_checkpoint(self):
        with tempfile.TemporaryDirectory() as d,patch.object(common,'STATE',Path(d)/'managed'):
            common.setup();protected=common.STATE/'models/champion.pt';protected.write_bytes(b'keep')
            old=common.STATE/'replay/old.gz';old.write_bytes(b'old')
            import os,time
            os.utime(old,(time.time()-8*86400,)*2)
            outside=Path(d)/'outside';outside.write_bytes(b'outside')
            (common.STATE/'replay/link.gz').symlink_to(outside)
            result=common.maintain_storage(reserve_gib=0)
            self.assertFalse(old.exists());self.assertTrue(protected.exists());self.assertTrue(outside.exists());self.assertTrue(result['ok'])
    def test_atomic_event_and_bundle(self):
        with tempfile.TemporaryDirectory() as d,patch.object(common,'STATE',Path(d)):
            common.setup();common.event('test','resource',time=123)
            self.assertEqual(common.read_json(Path(d)/'metrics/test.jsonl')['time'],123)
            p=Path(d)/'replay/test.gz';common.write_bundle(p,[{'result_wdl':None}])
            self.assertIsNone(common.read_bundle(p)[0]['result_wdl'])
class ReplayRegression(unittest.TestCase):
    def test_promotion_heads_and_unknown_result(self):
        import torch
        from autoloop.worker import targets,board_of
        from core.encoding import orient_move
        from core.moves import move_to_index
        b=chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')
        r=dict(start=b.fen(),fen=b.fen(),history=[],policy=[['a2a1q',.5],['a2a1n',.5]],teacher_policy=[['a2a1q',.5],['a2a1n',.5]],teacher_wdl=[0,.8,.2],result_wdl=None)
        with patch.object(torch.Tensor,'pin_memory',lambda t:t):x,p,pr,w=targets([r])
        self.assertAlmostEqual(float(p.sum()),1)
        self.assertAlmostEqual(float(pr[0,0]),.5);self.assertAlmostEqual(float(pr[0,3]),.5)
        idx=move_to_index(orient_move(chess.Move.from_uci('a2a1q'),b.turn))
        self.assertAlmostEqual(float(p[0,idx]),1)
        self.assertTrue(np.allclose(w.numpy(),[[0,.8,.2]]))
        self.assertEqual(board_of(r).fen(),b.fen())
    def test_dedup_preserves_rule_state(self):
        from autoloop.worker import sample_key
        a={'fen':chess.Board().fen(),'repetition':False}
        b=dict(a,fen=a['fen'].replace(' 0 1',' 99 1'))
        c=dict(a,repetition=True)
        self.assertNotEqual(sample_key(a),sample_key(b));self.assertNotEqual(sample_key(a),sample_key(c))


    def test_completed_selfplay_becomes_labelled_replay_rows(self):
        from autoloop.worker import training_rows
        board = chess.Board()
        summary = {
            'result': '1-0',
            'termination': 'CHECKMATE',
            'trajectory': [{
                'start': board.fen(), 'history': [], 'fen': board.fen(),
                'policy': [['e2e4', 1.0]], 'pieces': 32,
            }],
        }
        rows = training_rows(summary, source='model_crossplay', actor_role='small')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['teacher_policy'], [['e2e4', 1.0]])
        self.assertEqual(rows[0]['teacher_wdl'], [1, 0, 0])
        self.assertEqual(rows[0]['result_wdl'], [1, 0, 0])


class ReplayBufferRegression(unittest.TestCase):
    @staticmethod
    def _rows():
        board = chess.Board()
        rows = []
        piece_counts = (32, 24, 18, 12, 6)
        for stage, count in enumerate(piece_counts):
            for index in range(3):
                fen = board.fen().replace(" 0 1", f" {stage * 3 + index} 1")
                rows.append({
                    "fen": fen,
                    "start": fen,
                    "history": [],
                    "policy": [["e2e4", .9]],
                    "teacher_policy": [["d2d4", .9]],
                    "teacher_wdl": [.2, .6, .2],
                    "pieces": count,
                    "cp_loss_estimate": 40.0 if index == 0 else 0.0,
                })
        return rows

    def test_persistent_stage_balanced_reservoir(self):
        from autoloop.replay_buffer import ExperienceReplay, STAGE_NAMES, stage_for_row
        with tempfile.TemporaryDirectory() as d:
            with patch("autoloop.replay_buffer.event", lambda *args, **kwargs: None), \
                 patch("autoloop.replay_buffer.is_holdout", lambda fen: False):
                buffer = ExperienceReplay("layered", state_dir=Path(d),
                                          capacity=10, fresh_budget=5, reservoir_budget=5)
                rows = buffer.refresh(source_rows=self._rows())
                self.assertGreaterEqual(len(rows), 5)
                self.assertLessEqual(len(rows), 10)
                self.assertTrue(buffer.state_path.exists())
                self.assertEqual({stage_for_row(row) for row in rows}, set(STAGE_NAMES))
                retained = buffer.refresh(source_rows=[])
                self.assertEqual({stage_for_row(row) for row in retained}, set(STAGE_NAMES))

    def test_priority_and_target_version(self):
        from autoloop.replay_buffer import difficulty, replay_version
        rows = self._rows()
        easy = dict(rows[1], cp_loss_estimate=0.0,
                    policy=[["e2e4", .9]], teacher_policy=[["e2e4", .9]])
        hard = dict(rows[1], cp_loss_estimate=800.0, promotion_threat=True,
                    policy=[["e2e4", .9]], teacher_policy=[["d2d4", .9]],
                    teacher_wdl=[1.0, 0.0, 0.0], result_wdl=[0.0, 0.0, 1.0])
        self.assertGreater(difficulty(hard), difficulty(easy))
        self.assertNotEqual(replay_version([easy]),
                            replay_version([dict(easy, teacher_cp=100.0)]))


class LayeredRegression(unittest.TestCase):
    def test_stage_boundaries_are_contiguous(self):
        from autoloop.layered import STAGES, stage_for_piece_count, LAYERED_PARAMS
        self.assertGreaterEqual(LAYERED_PARAMS, 19_000_000)
        self.assertLess(LAYERED_PARAMS, 21_000_000)
        self.assertEqual([stage_for_piece_count(n) for n in range(32, 1, -1)],
                         [0] * 8 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 5)
        self.assertEqual(STAGES[-1].minimum, 2)

    def test_router_groups_and_restores_batch_order(self):
        from autoloop.layered import LayeredRouter
        def make(value):
            def evaluate(boards):
                n=len(boards)
                return (np.full((n,4096),value,np.float32),
                        np.full((n,4),value,np.float32),
                        np.full((n,3),value,np.float32))
            return evaluate
        router=LayeredRouter({i:make(i+1) for i in range(5)})
        boards=[chess.Board(),
                chess.Board('8/8/8/8/8/2k5/8/2K5 w - - 0 1'),
                chess.Board('8/8/8/8/8/8/2k5/2KQ4 w - - 0 1')]
        p,pr,w=router.evaluate_batch(boards)
        self.assertEqual(p[:,0].tolist(), [1.0,5.0,5.0])
        self.assertEqual(pr.shape,(3,4));self.assertEqual(w.shape,(3,3))

    def test_unknown_games_are_not_draws(self):
        from autoloop.evaluation import paired_game_scores, reliability, paired_sign_test
        games=[{'pair':0,'candidate_white':True,'result':'1-0'},
               {'pair':0,'candidate_white':False,'result':None},
               {'pair':1,'candidate_white':True,'result':'1-0'},
               {'pair':1,'candidate_white':False,'result':'0-1'}]
        self.assertEqual(paired_game_scores(games),[1.0])
        self.assertEqual(reliability(games)['unknown_games'],1)
        self.assertEqual(paired_sign_test([1.0])['positive_pairs'],1)

if __name__=='__main__':unittest.main()
