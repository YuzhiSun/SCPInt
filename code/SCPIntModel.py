import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt   # 新增这一行，放在其他 import 之后

class AnnDataProcessor:
    """
    处理 AnnData 输入，提取表达矩阵和批次标签
    支持 adata.X 为 dense 或 sparse 矩阵
    """
    def __init__(self, adata: ad.AnnData, batch_key: str = "batch"):
        self.adata      = adata
        self.batch_key  = batch_key
        self.le         = LabelEncoder()

        # 字符串批次 → 整数编码
        batch_str            = adata.obs[batch_key].values
        self.batch_int       = self.le.fit_transform(batch_str)   # np.ndarray
        self.batch_names     = list(self.le.classes_)             # 原始批次名列表
        self.n_batches       = len(self.batch_names)
        self.n_genes         = adata.n_vars

        print(f"Loaded {adata.n_obs} cells × {self.n_genes} proteins")
        print(f"Batches ({self.n_batches}): {self.batch_names}")


    def to_tensors(self):
        """返回 (X_tensor, batch_label_tensor)"""
        X = self.adata.X
        if hasattr(X, "toarray"):       # sparse → dense
            X = X.toarray()
        X_tensor     = torch.tensor(X, dtype=torch.float32)
        batch_tensor = torch.tensor(self.batch_int, dtype=torch.long)
        return X_tensor, batch_tensor

    def make_dataloader(self, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
        X_tensor, batch_tensor = self.to_tensors()
        dataset = TensorDataset(X_tensor, batch_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ─── 梯度反转层 ───────────────────────────────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class Encoder2(nn.Module):
    def __init__(self, n_genes: int, hidden_dims: list = [512, 256]):
        super().__init__()
        layers = []
        in_dim = n_genes
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        self.shared     = nn.Sequential(*layers)
        self.bio_head   = nn.Linear(in_dim, 128)
        self.batch_head = nn.Linear(in_dim, 64)

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([self.bio_head(h), self.batch_head(h)], dim=1)
        # return h

# ─── GMM Decoder ─────────────────────────────────────────────────────────────
class GMMDecoder(nn.Module):
    """
    zscore 输入数据有负值，mu 无需激活约束，sigma 用 softplus 保证正数
    """
    N_COMPONENTS = 2

    def __init__(self, n_genes: int, n_batches: int, hidden_dims: list = [256, 512]):
        super().__init__()
        self.n_genes = n_genes
        in_dim = 128 + n_batches
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        self.shared      = nn.Sequential(*layers)
        self.pi_head     = nn.Linear(in_dim, self.N_COMPONENTS)
        self.mu_heads    = nn.ModuleList([nn.Linear(in_dim, n_genes) for _ in range(self.N_COMPONENTS)])
        self.sigma_heads = nn.ModuleList([nn.Linear(in_dim, n_genes) for _ in range(self.N_COMPONENTS)])
        #
        # for i, head in enumerate(self.mu_heads):
        #     target_mu = 0.0 if i == 0 else 2.5
        #     nn.init.zeros_(head.weight)  # 权重置零 → 初始完全依赖 bias
        #     nn.init.constant_(head.bias, target_mu)
        #     print(f"GMM mu_heads[{i}] bias 已初始化为 {target_mu}")  # 调试确认

    def forward(self, bio_emb, batch_onehot):
        h     = self.shared(torch.cat([bio_emb, batch_onehot], dim=1))
        pi    = torch.softmax(self.pi_head(h), dim=-1)                      # [B, 2]
        mu    = torch.stack([head(h) for head in self.mu_heads], dim=1)     # [B, 2, G]
        sigma = torch.stack([
            torch.nn.functional.softplus(head(h)) + 1e-6
            for head in self.sigma_heads
        ], dim=1)                                                            # [B, 2, G]
        return pi, mu, sigma

# ─── GMM 重建损失 ─────────────────────────────────────────────────────────────
def gmm_reconstruction_loss(x, pi, mu, sigma):
    """
    负对数似然，对基因维度归一化避免高维累加导致数值爆炸
    x: [B, G], pi: [B, 2], mu: [B, 2, G], sigma: [B, 2, G]
    """
    x_expand = x.unsqueeze(1)                                               # [B, 1, G]
    log_prob  = (
        -0.5 * ((x_expand - mu) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x.device))
    ).sum(dim=-1) / x.shape[1]                                              # [B, 2] 除以G归一化

    log_mix = torch.logsumexp(torch.log(pi + 1e-8) + log_prob, dim=-1)     # [B]
    return -log_mix.mean()

def gmm_mixture_mean(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """计算 GMM 的混合均值（作为重建表达），用于 MSE 损失"""
    return (pi.unsqueeze(-1) * mu).sum(dim=1)  # [B, G]


# ─── Batch Classifier ────────────────────────────────────────────────────────
class BatchClassifier(nn.Module):
    def __init__(self, n_batches: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_batches)
        )

    def forward(self, x):
        return self.net(x)


# ─── Discriminator ───────────────────────────────────────────────────────────
class BatchDiscriminator(nn.Module):
    def __init__(self, n_batches: int, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(alpha)
        self.net = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_batches)
        )

    def forward(self, x):
        return self.net(self.grl(x))

    def set_alpha(self, alpha: float):
        self.grl.alpha = alpha

# ─── 完整模型 ─────────────────────────────────────────────────────────────────
class scProteoIntegrator(nn.Module):
    def __init__(self, n_genes: int, n_batches: int,
                 # enc_hidden: list = [512, 256, 128 + 64],
                 enc_hidden: list = [512, 256],
                 dec_hidden: list = [256, 512],
                 adv_alpha: float = 1.0):
        super().__init__()
        self.n_batches     = n_batches
        # self.encoder       = Encoder(n_genes, enc_hidden)
        self.encoder       = Encoder2(n_genes, enc_hidden)
        self.decoder       = GMMDecoder(n_genes, n_batches, dec_hidden)
        self.classifier    = BatchClassifier(n_batches)
        self.discriminator = BatchDiscriminator(n_batches, adv_alpha)

    def forward(self, x, batch_onehot):
        emb       = self.encoder(x)
        bio_emb   = emb[:, :128]
        batch_emb = emb[:, 128:]
        return (
            self.decoder(bio_emb, batch_onehot),
            self.classifier(batch_emb),
            self.discriminator(bio_emb),
            bio_emb,
            batch_emb,
        )


class IntegratorLoss(nn.Module):
    def __init__(self, lambda_cls: float = 1.0, lambda_adv: float = 1.0, lambda_mse: float = 1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_adv = lambda_adv
        self.lambda_mse = lambda_mse  # 新增：MSE 权重
        self.cls_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()  # 新增

    def forward(self, x, gmm_params, cls_logits, adv_logits, batch_labels):
        pi, mu, sigma = gmm_params
        l_recon = gmm_reconstruction_loss(x, pi, mu, sigma)

        # 新增：GMM 混合均值重建 + MSE
        recon_x = gmm_mixture_mean(pi, mu)
        l_mse = self.mse_loss(recon_x, x)

        l_cls = self.cls_loss(cls_logits, batch_labels)
        l_adv = self.adv_loss(adv_logits, batch_labels)

        total = (l_recon
                 + self.lambda_cls * l_cls
                 + self.lambda_adv * l_adv
                 + self.lambda_mse * l_mse)  # 新增 MSE 项

        return total, {
            "recon": l_recon.item(),
            "mse": l_mse.item(),  # 新增
            "cls": l_cls.item(),
            "adv": l_adv.item()
        }



# ─── 训练器 ───────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: scProteoIntegrator, lr: float = 1e-3,lr_reduc_factor=0.9,
                 lambda_cls: float = 1.0, lambda_adv: float = 1.0,
                 lambda_mse: float = 1.0,
                 device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.model     = model.to(self.device)
        self.criterion = IntegratorLoss(lambda_cls, lambda_adv, lambda_mse)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=lr_reduc_factor
        )
        # ─── 新增：损失历史记录，用于绘制曲线 ───────────────────────────────
        self.loss_history = {
            "epoch": [],
            "total": [],
            "recon": [],
            "mse": [],
            "cls": [],
            "adv": []
        }

    def _to_onehot(self, batch_labels: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(len(batch_labels), self.model.n_batches, device=self.device)
        onehot.scatter_(1, batch_labels.unsqueeze(1), 1.0)
        return onehot

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        totals = {"recon": 0.0, "cls": 0.0, "adv": 0.0, "total": 0.0, "mse": 0.0}

        for x, batch_labels in loader:
            x, batch_labels = x.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()
            gmm_params, cls_logits, adv_logits, _, _ = self.model(x, self._to_onehot(batch_labels))
            loss, loss_dict = self.criterion(x, gmm_params, cls_logits, adv_logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            totals["total"] += loss.item()
            for k, v in loss_dict.items():
                totals[k] += v

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        self.model.eval()
        return self.model.encoder(x.to(self.device))[:, :128].cpu(), self.model.encoder(x.to(self.device))[:,128:].cpu()

    def fit(self, loader: DataLoader, n_epochs: int = 100,
            adv_warmup_epochs: int = 10):
        for epoch in range(1, n_epochs + 1):
            alpha = 0.0 if epoch <= adv_warmup_epochs else min(
                1.0, (epoch - adv_warmup_epochs) / adv_warmup_epochs
            )
            self.model.discriminator.set_alpha(alpha)
            losses = self.train_epoch(loader)
            self.scheduler.step(losses["total"])
            # ─── 新增：每轮记录损失历史 ───────────────────────────────
            self.loss_history["epoch"].append(epoch)
            for k in ["total", "recon", "mse", "cls", "adv"]:
                self.loss_history[k].append(losses[k])
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | "
                      f"total={losses['total']:.4f} | "
                      f"recon={losses['recon']:.4f} | "
                      f"mse={losses['mse']:.4f} | "  
                      f"cls={losses['cls']:.4f} | "
                      f"adv={losses['adv']:.4f} |"
                      f"lr={self.optimizer.param_groups[0]['lr']:}"
                      )

    def save(self, path: str):
        """
        保存模型权重 + 重建模型所需的配置
        path: 文件路径，建议 .pt 后缀
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": {
                "n_genes": self.model.encoder.shared[0].in_features,
                "n_batches": self.model.n_batches,
                "enc_hidden": [l.out_features for l in self.model.encoder.shared if isinstance(l, nn.Linear)],
                "dec_hidden": [l.out_features for l in self.model.decoder.shared if isinstance(l, nn.Linear)],
            }
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto", **trainer_kwargs) -> "Trainer":
        """
        从文件恢复模型和训练器
        trainer_kwargs: lr / lambda_cls / lambda_adv 等，不传则用默认值
        """
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["model_config"]
        model = scProteoIntegrator(
            n_genes=cfg["n_genes"],
            n_batches=cfg["n_batches"],
            enc_hidden=cfg["enc_hidden"],
            dec_hidden=cfg["dec_hidden"],
        )
        trainer = cls(model, device=device, **trainer_kwargs)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Model loaded from {path}")
        return trainer

    def plot_losses(self, save_path: str = None, show: bool = True, figsize: tuple = (12, 6)):
        """
        绘制训练损失曲线
        - 主轴：total / recon / mse（数值通常较大）
        - 副轴：cls / adv（数值通常较小）
        """
        if not self.loss_history["epoch"]:
            print("⚠️ 尚未运行 fit()，无损失历史可绘制")
            return

        epochs = self.loss_history["epoch"]
        fig, ax = plt.subplots(figsize=figsize)

        # 主轴（total + recon + mse）
        ax.plot(epochs, self.loss_history["total"], label="Total Loss", color="black", linewidth=2.5)
        ax.plot(epochs, self.loss_history["recon"], label="GMM NLL (recon)", color="tab:blue")
        ax.plot(epochs, self.loss_history["mse"], label="MSE", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Main Loss")
        ax.grid(True, alpha=0.3)

        # 副轴（cls + adv）
        ax2 = ax.twinx()
        ax2.plot(epochs, self.loss_history["cls"], label="Classification", color="tab:green", linestyle="--")
        ax2.plot(epochs, self.loss_history["adv"], label="Adversarial", color="tab:red", linestyle="--")
        ax2.set_ylabel("Cls / Adv Loss")

        # 图例合并
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title("scProteoIntegrator Training Loss Curves")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"损失曲线已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
