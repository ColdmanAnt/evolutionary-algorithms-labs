import unittest
import numpy as np
from lab3 import (Gene, WIDTHS, accuracy, crossover, fit_scaler, make_data,
                  mutate, nondominated, predict_logits, random_gene, search,
                  train_model)


class TestLab3(unittest.TestCase):
    def test_all_architecture_parameter_counts(self):
        x, z, y, idx, *_ = make_data()
        for width in range(len(WIDTHS)):
            gene = Gene(7, width, 0)
            model = train_model(gene, z[idx['train']], y[idx['train']], 7, epochs=1)
            self.assertEqual(sum(a.size for a in model.values()), gene.parameters)

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
        a, history_a, _, _ = search(1,.004,25,z,y,idx,population_size=4)
        b, history_b, _, _ = search(1,.004,25,z,y,idx,population_size=4)
        for key in ['gene','fitness','evaluations','distinct_trainings']:
            self.assertEqual(a[key],b[key])
        self.assertEqual(len(history_a),25)
        self.assertEqual(a['evaluations'],25)
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
        changed_z[idx['test']]=1e6
        changed_y[idx['test']]=1-changed_y[idx['test']]
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

if __name__=='__main__':
    unittest.main()
