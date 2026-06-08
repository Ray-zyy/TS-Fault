import numpy as np

SEQ_LEN = 336
PRED_LEN = 96



class TimesFMModel:
    def __init__(self, device='cuda', context_len=SEQ_LEN, horizon_len=PRED_LEN,
                 per_core_batch_size=32):
        import timesfm
        self.horizon = horizon_len
        ctx_aligned = ((context_len + 31) // 32) * 32   # 336 -> 352
        backend = 'gpu' if str(device).startswith('cuda') else 'cpu'
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=per_core_batch_size,
                horizon_len=horizon_len,
                context_len=ctx_aligned,           # 352, 32的倍数
                num_layers=50,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

    def predict(self, x_context):

        B, L, C = x_context.shape
        out = np.zeros((B, self.horizon, C), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                s = [np.asarray(x_context[b, :, c], dtype=np.float32)]
                point, _ = self.tfm.forecast(s, freq=[0])
                out[b, :, c] = np.asarray(point)[0, :self.horizon]
        return out


class ChronosModel:
    def __init__(self, device='cuda', horizon_len=PRED_LEN,
                 model_id="amazon/chronos-bolt-base"):
        import torch
        from chronos import BaseChronosPipeline
        self.horizon = horizon_len
        self.torch = torch
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16,
        )

    def predict(self, x_context):
        B, L, C = x_context.shape
        flat = x_context.transpose(0, 2, 1).reshape(B * C, L)   # (B*C, L)
        flat = np.asarray(flat, dtype=np.float32)

        chunk = getattr(self, 'infer_batch_size', 256)  # 每批最多 256 条序列
        outs = []
        for s in range(0, flat.shape[0], chunk):
            ctx = self.torch.tensor(flat[s:s + chunk], dtype=self.torch.float32)
            with self.torch.no_grad():
                fc = self.pipe.predict(ctx, self.horizon)
            fc = np.asarray(fc.float().cpu()) if hasattr(fc, 'float') else np.asarray(fc)
            if fc.ndim == 3:                       # (chunk, Q, H) 取中位分位
                fc = fc[:, fc.shape[1] // 2, :]
            outs.append(fc[:, :self.horizon])
            del ctx
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

        med = np.concatenate(outs, axis=0)         # (B*C, H)
        return med.reshape(B, C, self.horizon).transpose(0, 2, 1)


class MoiraiModel:
    def __init__(self, device='cuda', context_len=SEQ_LEN, horizon_len=PRED_LEN,
                 size='base', patch_size=32, model_id="Salesforce/moirai-1.1-R-base"):
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.torch = torch
        self.device = device
        self.horizon = horizon_len
        module = MoiraiModule.from_pretrained(model_id)
        self.model = MoiraiForecast(
            module=module, prediction_length=horizon_len, context_length=context_len,
            patch_size=patch_size, num_samples=100, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        ).to(device).eval()

    def predict(self, x_context):
        B, L, C = x_context.shape
        torch = self.torch
        out = np.zeros((B, self.horizon, C), dtype=np.float32)
        with torch.no_grad():
            for c in range(C):
                past = torch.tensor(x_context[:, :, c:c+1], dtype=torch.float32, device=self.device)
                past_obs = torch.ones((B, L, 1), dtype=torch.bool, device=self.device)
                past_pad = torch.zeros((B, L), dtype=torch.bool, device=self.device)
                forecast = self.model(
                    past_target=past, past_observed_target=past_obs, past_is_pad=past_pad,
                )
                fc = forecast.cpu().numpy()
                if fc.ndim == 4:
                    fc = fc[..., 0]
                med = np.median(fc, axis=1)
                out[:, :, c] = med[:, :self.horizon]
        return out


def build_model(name, device='cuda'):
    name = name.lower()
    if name in ('timesfm', 'timesfm2', 'timesfm-2.0'):
        return TimesFMModel(device=device)
    if name in ('chronos', 'chronos-bolt', 'chronosbolt'):
        return ChronosModel(device=device)
    if name in ('moirai', 'moirai-1.1', 'moirai1.1'):
        return MoiraiModel(device=device)
    raise ValueError(f"未知模型: {name}")

FOUNDATION_MODELS = ['timesfm', 'chronos', 'moirai']