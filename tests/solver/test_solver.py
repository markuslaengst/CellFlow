import warnings

import jax
import numpy as np
import optax
import pytest

import cellflow
from cellflow._compat import ConstantNoiseFlow
from cellflow.solvers import _genot, _otfm, _otrfm
from cellflow.utils import match_linear

src = {
    ("drug_1",): np.random.rand(10, 5),
    ("drug_2",): np.random.rand(10, 5),
}
cond = {
    ("drug_1",): {"drug": np.random.rand(1, 1, 3)},
    ("drug_2",): {"drug": np.random.rand(1, 1, 3)},
}
vf_rng = jax.random.PRNGKey(111)


class TestSolver:
    @pytest.mark.parametrize("solver_class", ["otfm", "genot"])
    def test_predict_batch(self, dataloader, solver_class):
        if solver_class == "otfm":
            vf_class = cellflow.networks.ConditionalVelocityField
        else:
            vf_class = cellflow.networks.GENOTConditionalVelocityField

        opt = optax.adam(1e-3)
        vf = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        if solver_class == "otfm":
            solver = _otfm.OTFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )
        else:
            solver = _genot.GENOT(
                vf=vf,
                data_match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                source_dim=5,
                target_dim=5,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=1,
        )
        x_pred_batched = solver.predict(src, cond)

        x_pred_nonbatched = jax.tree.map(
            solver.predict,
            src,
            cond,  # type: ignore[attr-defined]
        )

        assert x_pred_batched[("drug_1",)].shape == x_pred_nonbatched[("drug_1",)].shape
        assert np.allclose(
            x_pred_batched[("drug_1",)],
            x_pred_nonbatched[("drug_1",)],
            atol=1e-1,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("solver_class", ["otfm", "genot"])
    def test_predict_batched_deprecation_warning(self, dataloader, solver_class):
        if solver_class == "otfm":
            vf_class = cellflow.networks.ConditionalVelocityField
        else:
            vf_class = cellflow.networks.GENOTConditionalVelocityField

        opt = optax.adam(1e-3)
        vf = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        if solver_class == "otfm":
            solver = _otfm.OTFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )
        else:
            solver = _genot.GENOT(
                vf=vf,
                data_match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                source_dim=5,
                target_dim=5,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=1,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.predict(src, cond, batched=True)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "batched" in str(w[0].message)

    @pytest.mark.parametrize("ema", [0.5, 1.0])
    def test_EMA(self, dataloader, ema):
        vf_class = cellflow.networks.ConditionalVelocityField
        drug = np.random.rand(2, 1, 3)
        opt = optax.adam(1e-3)
        vf1 = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(6, 6),
            decoder_dims=(5, 5),
        )

        solver1 = _otfm.OTFlowMatching(
            vf=vf1,
            match_fn=match_linear,
            probability_path=ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions={"drug": drug},
            rng=vf_rng,
            ema=ema,
        )
        trainer1 = cellflow.training.CellFlowTrainer(solver=solver1)
        trainer1.train(
            dataloader=dataloader,
            num_iterations=5,
            valid_freq=10,
        )

        if ema == 1.0:
            assert jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.allclose(x, y, atol=1e-5, rtol=1e-2),
                    solver1.vf_state_inference.params,
                    solver1.vf_state.params,
                )
            )
        else:
            assert not jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.allclose(x, y, atol=1e-5),
                    solver1.vf_state_inference.params,
                    solver1.vf_state.params,
                )
            )

    def test_otrfm_requires_fitted_nystroem(self):
        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(6, 6),
            decoder_dims=(5, 5),
        )

        with pytest.raises(TypeError, match="fitted SpectralNystroem-like instance"):
            _otrfm.OTRFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=optax.adam(1e-3),
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )

    @pytest.mark.parametrize(
        ("loss_mode", "trajectory_end_time"), [("final", None), ("trajectory", None), ("trajectory", 0.99)]
    )
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    def test_otrfm_loss_modes_train(self, loss_mode, trajectory_end_time, condition_mode):
        import jax.numpy as jnp

        class EuclideanPremetric:
            def distance_grad_and_norm_batch(self, xt, x1):
                diff = xt - x1
                d = jnp.linalg.norm(diff, axis=1)
                grad = diff / jnp.maximum(d[:, None], 1e-12)
                return d, grad, jnp.sum(grad * grad, axis=1)

        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_mode=condition_mode,
            condition_embedding_dim=12,
            hidden_dims=(16, 16),
            decoder_dims=(16, 16),
        )
        solver = _otrfm.OTRFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=ConstantNoiseFlow(0.0),
            optimizer=optax.adam(1e-3),
            conditions={"drug": np.random.rand(2, 1, 3)},
            rng=vf_rng,
            nystroem=EuclideanPremetric(),
            interpolation_steps=4,
            loss_mode=loss_mode,
            trajectory_end_time=trajectory_end_time,
        )
        batch = {
            "src_cell_data": jnp.asarray(np.random.rand(8, 5)),
            "tgt_cell_data": jnp.asarray(np.random.rand(8, 5)) + 3.0,
            "condition": {"drug": jnp.asarray(np.random.rand(1, 1, 3))},
        }
        losses = [float(solver.step_fn(jax.random.PRNGKey(i), batch)) for i in range(3)]
        assert np.all(np.isfinite(losses))
        assert int(solver.vf_state.step) == 3

    def test_otrfm_rejects_unknown_loss_mode(self):
        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5, max_combination_length=2, condition_embedding_dim=12,
            hidden_dims=(6, 6), decoder_dims=(5, 5),
        )

        class Premetric:
            def distance_grad_and_norm_batch(self, xt, x1):
                raise NotImplementedError

        with pytest.raises(ValueError, match="loss_mode"):
            _otrfm.OTRFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=optax.adam(1e-3),
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
                nystroem=Premetric(),
                loss_mode="all",
            )
