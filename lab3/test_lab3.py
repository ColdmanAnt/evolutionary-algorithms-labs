import unittest
from unittest.mock import patch
from pathlib import Path
import shutil
import tempfile
import numpy as np
from lab3 import (Gene, WIDTHS, EPOCHS, accuracy, crossover, fit_scaler, make_data,
                  mutate, nondominated, predict_logits, random_gene, search,
                  select_representatives, train_model, run_experiment)
from audit_saved import training_gradient_oracle, verify_saved


class TestLab3(unittest.TestCase):
    def test_all_architecture_parameter_counts(self):
        x, z, y, idx, *_ = make_data()
        genes = {Gene(mask, width, activation).canonical()
                 for mask in range(1, 16) for width in range(len(WIDTHS)) for activation in (0, 1)}
        self.assertEqual(len(genes),135)
        for gene in genes:
            model = train_model(gene, z[idx['train']], y[idx['train']], 7, epochs=0)
            self.assertEqual(sum(a.size for a in model.values()), gene.parameters)

    def test_training_gradients_by_independent_finite_differences(self):
        result = training_gradient_oracle()
        self.assertEqual(result['finite_difference_coordinates'],54)
        self.assertLess(result['max_absolute_gradient_error'],5e-8)

    def test_variation_syntax(self):
        rng = np.random.default_rng(33)
        for _ in range(1000):
            gene = mutate(crossover(random_gene(rng), random_gene(rng), rng), rng)
            self.assertTrue(gene.valid())
            self.assertEqual(gene, gene.canonical())

    def test_training_reproducibility(self):
        _, z, y, idx, *_ = make_data()
        gene=Gene(7, 3, 1)
        a=train_model(gene,z[idx['train']],y[idx['train']],55,epochs=10)
        b=train_model(gene,z[idx['train']],y[idx['train']],55,epochs=10)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])

    def test_search_budget_reproducibility(self):
        _, z, y, idx, *_ = make_data()
        a, history_a, _, _ = search(1,.004,26,z,y,idx,population_size=4)
        immutable_idx={name:tuple(values) for name,values in idx.items()}
        b, history_b, _, _ = search(1,.004,26,z,y,immutable_idx,population_size=4)
        for key in ['gene','fitness','evaluations','distinct_trainings']:
            self.assertEqual(a[key],b[key])
        self.assertEqual(len(history_a),26)
        self.assertEqual(a['evaluations'],26)
        self.assertEqual(history_a,history_b)
        self.assertTrue(all(x['best_fitness']>=y['best_fitness'] for x,y in zip(history_a,history_a[1:])))

    def test_scaler_and_split_leakage(self):
        x,z,y,idx,mean,scale=make_data()
        np.testing.assert_allclose(mean,x[idx['train']].mean(axis=0))
        np.testing.assert_allclose(scale,x[idx['train']].std(axis=0))
        self.assertEqual(len(set(np.concatenate(list(idx.values())))),len(x))
        changed=x.copy()
        changed[idx['test']]=1e8
        new_mean,new_scale=fit_scaler(changed[idx['train']])
        np.testing.assert_array_equal(mean,new_mean)
        np.testing.assert_array_equal(scale,new_scale)

    def test_search_is_independent_of_test_features_and_labels(self):
        _,z,y,idx,*_=make_data()
        baseline,*_=search(22,.004,4,z,y,idx,population_size=4)
        changed_z,changed_y=z.copy(),y.copy()
        # Даже отсутствующие test-значения не должны влиять на поиск.
        changed_z[idx['test']]=np.nan
        changed_y[idx['test']]=np.nan
        changed,*_=search(22,.004,4,changed_z,changed_y,idx,population_size=4)
        self.assertEqual(baseline['gene'],changed['gene'])
        self.assertEqual(baseline['fitness'],changed['fitness'])

    def test_nondomination(self):
        records=[{'validation_loss':.1,'parameters':10},
                 {'validation_loss':.2,'parameters':5},
                 {'validation_loss':.3,'parameters':8}]
        result=nondominated(records)
        self.assertEqual(len(result),2)
        self.assertNotIn(records[2],result)

    def test_archive_matches_pairwise_oracle_with_ties_and_permutations(self):
        rng=np.random.default_rng(1106)
        records=[{'id':i,'validation_loss':float(loss),'parameters':int(size)}
                 for i,(loss,size) in enumerate(zip(rng.integers(1,12,50)/10,rng.integers(1,12,50)))]
        records.extend([{'id':50,'validation_loss':.05,'parameters':15},
                        {'id':51,'validation_loss':.05,'parameters':15}])
        values=np.array([[r['validation_loss'],r['parameters']] for r in records])
        expected={r['id'] for i,r in enumerate(records)
                  if not np.any(np.all(values<=values[i],axis=1)&np.any(values<values[i],axis=1))}
        self.assertEqual({r['id'] for r in nondominated(records)},expected)
        shuffled=[records[i] for i in rng.permutation(len(records))]
        transformed=[dict(r,validation_loss=2*r['validation_loss']+3) for r in shuffled]
        self.assertEqual({r['id'] for r in nondominated(transformed)},expected)

    def test_representative_selection_is_independent_of_test_scores(self):
        records=[{'parameters':p,'validation_loss':1/p,'test_loss':p,'test_accuracy':1/p}
                 for p in [2,4,8,16,32]]
        chosen=select_representatives(records)
        corrupted=[dict(r,test_loss=float('nan'),test_accuracy=99) for r in records]
        self.assertEqual([r['parameters'] for r in chosen],
                         [r['parameters'] for r in select_representatives(corrupted)])
        self.assertEqual([r['parameters'] for r in chosen],[2,8,32])
        with self.assertRaisesRegex(ValueError,'increase --runs or --budget'):
            select_representatives(records[:2])

    def test_cache_counts_candidate_calls_and_actual_trainings_separately(self):
        _,z,y,idx,*_=make_data()
        fixed=Gene(3,1,0)
        with patch('lab3.random_gene',return_value=fixed), patch('lab3.crossover',return_value=fixed), \
             patch('lab3.mutate',return_value=fixed), patch('lab3.train_model',wraps=train_model) as trainer:
            result,history,cache,_=search(21,.004,17,z,y,idx,population_size=4)
        self.assertEqual(result['evaluations'],17)
        self.assertEqual(len(history),17)
        self.assertEqual(result['distinct_trainings'],1)
        self.assertEqual(result['training_steps'],EPOCHS)
        self.assertEqual(trainer.call_count,1)
        self.assertEqual(len(cache),1)

    def test_invalid_genes_and_training_inputs_raise_value_error(self):
        x=np.ones((3,4));y=np.array([0.,1.,0.])
        for gene in [Gene(1.5,0,0),Gene(1,1.5,0),Gene(True,0,0),Gene(1,0,.0),Gene(0,0,0)]:
            self.assertFalse(gene.valid())
            with self.assertRaises(ValueError):train_model(gene,x,y,1)
        for kwargs in [{'epochs':.5},{'epochs':-1},{'run_seed':-1}]:
            options={'run_seed':1,**kwargs}
            with self.assertRaises(ValueError):train_model(Gene(1,0,0),x,y,**options)
        with self.assertRaises(ValueError):train_model(Gene(1,0,0),x,np.ones((3,1)),1)
        with self.assertRaises(ValueError):fit_scaler(np.empty((0,4)))

    def test_invalid_search_inputs_and_disjoint_split_validation(self):
        _,z,y,idx,*_=make_data()
        for kwargs in [{'budget':24.5},{'seed':1.5},{'penalty':float('nan')},
                       {'penalty':-.1},{'population_size':4.5}]:
            options={'seed':1,'penalty':.004,'budget':24,**kwargs}
            with self.assertRaises(ValueError):search(z=z,y=y,indices=idx,**options)
        overlap={k:v.copy() for k,v in idx.items()}
        overlap['validation'][0]=overlap['train'][0]
        with self.assertRaisesRegex(ValueError,'disjoint'):search(1,.004,24,z,y,overlap)
        with self.assertRaises(ValueError):nondominated([{'validation_loss':float('nan'),'parameters':3}])
        with self.assertRaises(ValueError):run_experiment('must-not-be-created',runs=0)

    def test_saved_model_corruption_is_detected(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='.audit-test-',dir=root) as temporary:
            copied=Path(temporary)/'bundle'
            shutil.copytree(root,copied,ignore=shutil.ignore_patterns('.audit-test-*','__pycache__'))
            with np.load(copied/'model_1.npz',allow_pickle=False) as archive:
                weights={k:archive[k].copy() for k in archive.files}
            weights['w'][0]+=.5
            np.savez(copied/'model_1.npz',**weights)
            with self.assertRaisesRegex(ValueError,'model_1.npz: validation_loss'):
                verify_saved(directory=copied)

if __name__=='__main__':
    unittest.main()
