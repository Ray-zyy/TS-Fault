import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import weight_norm
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    class _DummyModule:
        pass
    class nn:  # type: ignore
        Module = _DummyModule
        class ModuleList: pass
        class Linear: pass
        class Conv1d: pass
        class ReLU: pass
        class Dropout: pass
        class Sequential: pass
        class Parameter: pass
        class LSTM: pass
        class GRU: pass


def _instance_norm(x_enc):
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x = x / stdev
    return x, means, stdev


def _instance_denorm(y, means, stdev, pred_len):
    y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
    y = y + means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
    return y


class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        hidden = configs.d_model
        dropout = configs.dropout

        layers = [nn.Linear(self.seq_len * self.enc_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, configs.e_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, self.pred_len * self.c_out)]
        self.net = nn.Sequential(*layers)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x, means, stdev = _instance_norm(x_enc)
        B = x.size(0)
        y = self.net(x.reshape(B, -1)).reshape(B, self.pred_len, self.c_out)
        return _instance_denorm(y, means, stdev, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]


class _RNNBase(nn.Module):
    def __init__(self, configs, rnn_cls):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.rnn = rnn_cls(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(configs.d_model, self.pred_len * self.c_out)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x, means, stdev = _instance_norm(x_enc)
        out, _ = self.rnn(x)            # [B, L, d_model]
        h = out[:, -1]                 # 取最后时刻 [B, d_model]
        y = self.proj(h).reshape(-1, self.pred_len, self.c_out)
        return _instance_denorm(y, means, stdev, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]


class LSTM(_RNNBase):
    def __init__(self, configs):
        super().__init__(configs, nn.LSTM)


class GRU(_RNNBase):
    def __init__(self, configs):
        super().__init__(configs, nn.GRU)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """locuslab/TCN verbatim: weight_norm + dilated Conv1d + ReLU + dropout, 残差连接"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        kernel_size = getattr(configs, 'kernel_size', 3)
        num_channels = [configs.d_model] * configs.e_layers
        self.tcn = TemporalConvNet(configs.enc_in, num_channels,
                                   kernel_size=kernel_size, dropout=configs.dropout)
        self.proj = nn.Linear(num_channels[-1], self.pred_len * self.c_out)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x, means, stdev = _instance_norm(x_enc)
        h = x.transpose(1, 2)          # [B, C, L]
        h = self.tcn(h)                # [B, d_model, L]
        last = h[:, :, -1]             # [B, d_model]
        y = self.proj(last).reshape(-1, self.pred_len, self.c_out)
        return _instance_denorm(y, means, stdev, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]


class NLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.individual = getattr(configs, 'individual', False)

        if self.individual:
            self.Linear = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = x_enc                       # [B, L, C]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            out = torch.zeros([x.size(0), self.pred_len, x.size(2)],
                              dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, :, i] = self.Linear[i](x[:, :, i])
            x = out
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x[:, :, :self.c_out]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]


class NBeatsBlock(nn.Module):
    """ServiceNow/N-BEATS verbatim: 全连接堆栈 + basis_parameters 线性层 + 基函数"""
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)] +
            [nn.Linear(in_features=layer_size, out_features=layer_size)
             for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, input_mask):
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial, backcast_size, forecast_size):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(
                np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :]
                for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(
                np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :]
                for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)

    def forward(self, theta):
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics, backcast_size, forecast_size):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=float),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=float) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=float)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=float)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
            requires_grad=False)
        self.backcast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
            requires_grad=False)
        self.forecast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
            requires_grad=False)
        self.forecast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
            requires_grad=False)

    def forward(self, theta):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum(
            'bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
            self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum(
            'bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = torch.einsum(
            'bp,pt->bt', theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum(
            'bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
            self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast


class NBEATS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        stack_types = getattr(configs, 'stack_types', ['generic', 'generic', 'generic'])
        layer_size = configs.d_model
        n_layers = 4

        blocks = []
        for stype in stack_types:
            if stype == 'generic':
                basis = GenericBasis(self.seq_len, self.pred_len)
                theta_size = self.seq_len + self.pred_len
            elif stype == 'trend':
                degree = getattr(configs, 'trend_degree', 2)
                basis = TrendBasis(degree, self.seq_len, self.pred_len)
                theta_size = 2 * (degree + 1)
            else:  # seasonality
                harmonics = getattr(configs, 'harmonics', 1)
                basis = SeasonalityBasis(harmonics, self.seq_len, self.pred_len)
                theta_size = 4 * int(np.ceil(harmonics / 2 * self.pred_len) - (harmonics - 1))
            blocks.append(NBeatsBlock(self.seq_len, theta_size, basis, n_layers, layer_size))
        self.nbeats = NBeats(nn.ModuleList(blocks))

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # channel-independent: [B,L,C] -> [B*C, L]
        B, L, C = x_enc.shape
        x = x_enc.permute(0, 2, 1).reshape(B * C, L)
        mask = torch.ones_like(x)
        y = self.nbeats(x, mask)                       # [B*C, pred_len]
        y = y.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]
        return y[:, :, :self.c_out]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]


class NaiveModel:
    is_stat = True
    def __init__(self, configs):
        self.pred_len = configs.pred_len
    def fit(self, x_clean, y_target):
        pass  # 无参数
    def predict(self, x):
        return np.repeat(x[:, -1:, :], self.pred_len, axis=1)


class SeasonalNaiveModel:
    is_stat = True
    def __init__(self, configs, season=24):
        self.pred_len = configs.pred_len
        self.season = season
    def fit(self, x_clean, y_target):
        pass
    def predict(self, x):
        B, L, C = x.shape
        s = min(self.season, L)
        pattern = x[:, -s:, :]
        n_repeat = (self.pred_len // s) + 1
        return np.tile(pattern, (1, n_repeat, 1))[:, :self.pred_len, :]



class ARIMAModel:
    """每个 (sample, channel) 独立拟合 ARIMA(p,d,q)。慢, 建议跳过 Electricity。"""
    is_stat = True
    def __init__(self, configs, order=(2, 1, 2), max_iter=20):
        self.pred_len = configs.pred_len
        self.order = order
        self.max_iter = max_iter
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self._ARIMA = ARIMA
        except ImportError:
            self._ARIMA = None
    def fit(self, x_clean, y_target):
        pass  # 在 predict 时即时拟合 (statsmodels 设计)
    def predict(self, x):
        if self._ARIMA is None:
            return np.repeat(x[:, -1:, :], self.pred_len, axis=1)
        B, L, C = x.shape
        out = np.zeros((B, self.pred_len, C), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                ts = x[b, :, c]
                if np.std(ts) < 1e-6:
                    out[b, :, c] = ts[-1]; continue
                try:
                    m = self._ARIMA(ts, order=self.order)
                    fit = m.fit(method_kwargs={'maxiter': self.max_iter, 'disp': 0})
                    out[b, :, c] = fit.forecast(steps=self.pred_len)
                except Exception:
                    out[b, :, c] = ts[-1]
        return out


class VARModel:
    is_stat = True
    def __init__(self, configs, maxlags=10):
        self.pred_len = configs.pred_len
        self.maxlags = maxlags
        try:
            from statsmodels.tsa.api import VAR
            self._VAR = VAR
        except ImportError:
            self._VAR = None
    def fit(self, x_clean, y_target):
        pass
    def predict(self, x):
        if self._VAR is None:
            return np.repeat(x[:, -1:, :], self.pred_len, axis=1)
        B, L, C = x.shape
        out = np.zeros((B, self.pred_len, C), dtype=np.float32)
        for b in range(B):
            X = x[b]
            if C < 2 or np.std(X) < 1e-6:
                out[b] = np.repeat(X[-1:, :], self.pred_len, axis=0); continue
            try:
                model = self._VAR(X)
                lag = min(self.maxlags, max(1, L // (C * 5)))
                fit = model.fit(maxlags=lag, verbose=False)
                out[b] = fit.forecast(X[-fit.k_ar:], steps=self.pred_len)
            except Exception:
                out[b] = np.repeat(X[-1:, :], self.pred_len, axis=0)
        return out


class ETSModel:
    is_stat = True
    def __init__(self, configs, seasonal_periods=24):
        self.pred_len = configs.pred_len
        self.seasonal_periods = seasonal_periods
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self._ETS = ExponentialSmoothing
        except ImportError:
            self._ETS = None
    def fit(self, x_clean, y_target):
        pass
    def predict(self, x):
        if self._ETS is None:
            return np.repeat(x[:, -1:, :], self.pred_len, axis=1)
        B, L, C = x.shape
        out = np.zeros((B, self.pred_len, C), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                ts = x[b, :, c]
                if np.std(ts) < 1e-6:
                    out[b, :, c] = ts[-1]; continue
                try:
                    m = self._ETS(ts, trend='add', seasonal=None)
                    fit = m.fit()
                    out[b, :, c] = fit.forecast(self.pred_len)
                except Exception:
                    out[b, :, c] = ts[-1]
        return out

class _TreeForecaster:
    is_stat = True
    def __init__(self, configs, kind='xgb', n_lags=48):
        self.pred_len = configs.pred_len
        self.n_lags = n_lags
        self.kind = kind
        self.models = None
        try:
            if kind == 'xgb':
                from xgboost import XGBRegressor
                self._make = lambda: XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                    n_jobs=4, verbosity=0)
            else:
                from sklearn.ensemble import RandomForestRegressor
                self._make = lambda: RandomForestRegressor(
                    n_estimators=150, max_depth=None, n_jobs=4)
            from sklearn.multioutput import MultiOutputRegressor
            self._wrap = MultiOutputRegressor
            self._ok = True
        except ImportError:
            self._ok = False

    def _build_xy(self, x_clean, y_target):
        B, L, C = x_clean.shape
        H = y_target.shape[1]
        feats, targs = [], []
        for b in range(B):
            for c in range(C):
                # 用最后 n_lags 步当特征, 预测 H 步
                feats.append(x_clean[b, -self.n_lags:, c])
                targs.append(y_target[b, :, c])
        return np.array(feats), np.array(targs)

    def fit(self, x_clean, y_target):
        if not self._ok:
            return
        X, Y = self._build_xy(x_clean, y_target)
        self.model = self._wrap(self._make())
        self.model.fit(X, Y)

    def predict(self, x):
        if not self._ok or not hasattr(self, 'model'):
            return np.repeat(x[:, -1:, :], self.pred_len, axis=1)
        B, L, C = x.shape
        out = np.zeros((B, self.pred_len, C), dtype=np.float32)
        for b in range(B):
            feats = np.array([x[b, -self.n_lags:, c] for c in range(C)])  # [C, n_lags]
            preds = self.model.predict(feats)                              # [C, H]
            out[b] = preds.T
        return out


class XGBoostModel(_TreeForecaster):
    def __init__(self, configs):
        super().__init__(configs, kind='xgb')


class RandomForestModel(_TreeForecaster):
    def __init__(self, configs):
        super().__init__(configs, kind='rf')


DL_MODELS = {
    'MLP': MLP,
    'LSTM': LSTM,
    'GRU': GRU,
    'TCN': TCN,
    'NLinear': NLinear,
    'NBEATS': NBEATS,
}

STAT_MODELS = {
    'Naive': NaiveModel,
    'SeasonalNaive': SeasonalNaiveModel,
    'ARIMA': ARIMAModel,
    'VAR': VARModel,
    'ETS': ETSModel,
    'XGBoost': XGBoostModel,
    'RandomForest': RandomForestModel,
}

ALL_MODELS = {**DL_MODELS, **STAT_MODELS}


def build_model(name, configs):
    if name not in ALL_MODELS:
        raise ValueError(f"未知模型 {name}, 可选: {list(ALL_MODELS)}")
    return ALL_MODELS[name](configs)
