# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
JEPAMatch: Geometric Representation Shaping for Semi-Supervised Learning.
https://github.com/aah94/JEPAMatch

Curriculum Level (pseudo-label selection) uses only the weak and strong views
(FlexMatch's class-adaptive confidence thresholding: weak view -> pseudo-label
+ confidence, strong view -> masked consistency-loss target). Representation
Level adds a JEPA-style prediction loss over weak/strong/local views plus
Adaptive Class-wise SIGReg (see semilearn/nets/jepa_modules.py), which forms
per-class isotropic Gaussians in the projector's latent space instead of a
single global one, with an Active Repulsion Loss keeping class means apart.

The paper defines exactly three view types -- Weak, Strong, Local -- and its
own text calls weak+strong "the global views" (Sec. 4; Fig. 1 shows only these
three, no separate global-augmented branch). This repo used to additionally
augment its own dedicated pair of "global" crops (x_ulb_g / num_global) for
the representation loss; that branch cost two extra forward passes per step
for views the paper never describes, so it's gone -- "global" below is simply
the weak and strong views already computed for the Curriculum Level.

Two design choices differ from the vanilla LeJEPA reference this builds on
(arXiv:2511.08544) and are independently configurable -- see `centroid_mean`
and `sigreg_views` in `get_argument()`:
  - LeJEPA's prediction target is the mean of its global views; this repo's
    JEPAMatch paper instead targets the mean of local views (`centroid_mean`).
  - LeJEPA applies SIGReg to all views uniformly; this repo's paper applies
    it to local views only, reasoning that the prediction loss already pulls
    weak/strong toward the same distribution (`sigreg_views`).
"""
import torch

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .utils import FlexMatchThresholdingHook, JEPADiagnosticsHook, get_alpha_schedule


@ALGORITHMS.register("jepamatch")
class JEPAMatch(AlgorithmBase):
    """
    JEPAMatch algorithm.

    Args (beyond FlexMatch's T / p_cutoff / hard_label / thresh_warmup):
        - beta (`float`): SIGReg balance within the representation loss.
            L_rep = (1 - beta) * L_pred + beta * L_SIGReg [+ repulsion_coef * L_repulsion in the main phase]
        - lambda_rep (`float`): weight of the representation loss in the total loss.
        - repulsion_coef (`float`, default 1000.0): multiplier on the repulsion term.
            IMPLEMENTATION NOTE (paper discrepancy, disclosed here rather than hidden): the
            paper's L_repulsion equation carries no explicit weight, but the code that produced
            every reported result always multiplied it by 1000 -- L_repulsion is a bounded,
            relu-gated cosine-similarity quantity (naturally ~1e-5 after relu zeroes ~95% of
            pairs and squaring crushes the rest), while L_pred and L_SIGReg are extensive in
            feature dimension and batch size respectively (naturally ~0.3-2.5). Without this
            weight, repulsion is numerically inert; 1000 is the default so this code reproduces
            the paper's tables out of the box. See semilearn/nets/jepa_modules.py for the
            underlying AdaptiveSIGReg.forward.
        - warmup_ratio (`float`): fraction of total training iterations spent in the
            SIGReg warmup phase before class-wise shaping begins (paper's T_warm / T).
        - centroid_mean (`str`, "local" | "global"): which views' average is the JEPA
            prediction-loss target (every view is then regressed toward it).
            "local"  = mean of the K local-view projections.
            "global" = mean of the global (weak + strong) view projections.
        - sigreg_views (`str`, "local" | "all"): which projections (Adaptive Class-wise)
            SIGReg regularizes.
            "local" = local views only (this paper's stated design: weak/strong
                      "implicitly inherit" normality via the prediction loss).
            "all"   = every view: weak + strong + local.
    """

    CENTROID_MEAN_CHOICES = ("local", "global")
    SIGREG_VIEWS_CHOICES = ("local", "all")

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(
            T=args.T,
            p_cutoff=args.p_cutoff,
            beta=args.beta,
            lambda_rep=args.lambda_rep,
            warmup_ratio=args.warmup_ratio,
            hard_label=args.hard_label,
            centroid_mean=args.centroid_mean,
            sigreg_views=args.sigreg_views,
            repulsion_coef=args.repulsion_coef,
        )

    def init(self, T, p_cutoff, beta, lambda_rep, warmup_ratio=0.25,
             hard_label=True, centroid_mean="local", sigreg_views="local", repulsion_coef=1000.0):
        assert centroid_mean in self.CENTROID_MEAN_CHOICES, \
            f"centroid_mean must be one of {self.CENTROID_MEAN_CHOICES}, got {centroid_mean!r}"
        assert sigreg_views in self.SIGREG_VIEWS_CHOICES, \
            f"sigreg_views must be one of {self.SIGREG_VIEWS_CHOICES}, got {sigreg_views!r}"
        self.T = T
        self.p_cutoff = p_cutoff
        self.beta = beta
        self.lambda_rep = lambda_rep
        self.warmup_ratio = warmup_ratio
        self.use_hard_label = hard_label
        self.centroid_mean = centroid_mean
        self.sigreg_views = sigreg_views
        self.repulsion_coef = repulsion_coef
        self.last_train_info = None

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            FlexMatchThresholdingHook(
                ulb_dest_len=self.args.ulb_dest_len,
                num_classes=self.num_classes,
                thresh_warmup=self.args.thresh_warmup,
            ),
            "MaskingHook",
        )
        if getattr(self.args, "save_train_info", False):
            self.register_hook(JEPADiagnosticsHook(), "JEPADiagnosticsHook")
        super().set_hooks()

    def _representation_loss(self, x_ulb_l, proj_weak, proj_strong, proj_x_lb,
                              pseudo_label, mask, y_lb, x_ulb_g=None):
        """Representation Level: JEPA prediction loss + Adaptive Class-wise SIGReg.

        By default (x_ulb_g=None), the paper's "global views" are the weak and
        strong views themselves (Sec. 4) -- reused here from the Curriculum
        Level forward pass rather than augmenting a separate global-view
        branch, which would cost extra inference for views the paper doesn't
        describe (see the module docstring).

        If the dataset instead provides `x_ulb_g` (an independently-augmented
        global-crop batch, config: global_source=separate), the representation
        loss runs entirely on {global, local} views and weak/strong are used
        for pseudo-labeling ONLY -- fully decoupling the two branches, an
        experimental alternative to reusing weak/strong.

        Every view is a fixed-size (B, D) block, so both losses below just
        pick which blocks to concatenate along dim 0 -- `torch.cat(...).view
        (num_rep, B, D)` recovers the per-view structure.
        """
        b = proj_weak.shape[0]
        proj_local = self.model(torch.cat(x_ulb_l, dim=0))["proj"]

        if x_ulb_g is not None:
            proj_global = self.model(torch.cat(x_ulb_g, dim=0))["proj"]
            all_views = torch.cat([proj_global, proj_local], dim=0)         # weak/strong excluded: pseudo-labeling only
            global_views = proj_global
        else:
            all_views = torch.cat([proj_weak, proj_strong, proj_local], dim=0)
            global_views = torch.cat([proj_weak, proj_strong], dim=0)       # weak + strong double as "global"

        # Prediction loss: every view regresses toward the chosen centroid.
        centroid_source = proj_local if self.centroid_mean == "local" else global_views
        centroid = centroid_source.view(-1, b, centroid_source.size(-1)).mean(dim=0)
        centroid_target = centroid.repeat(all_views.size(0) // b, 1)
        pred_loss = (all_views - centroid_target).pow(2).sum(dim=-1).mean()

        # Adaptive Class-wise SIGReg, degenerates to plain global SIGReg during warmup (alpha=0).
        sigreg_input = proj_local if self.sigreg_views == "local" else all_views
        alpha = get_alpha_schedule(self.it, self.num_train_iter, self.warmup_ratio)
        model = self.model.module if hasattr(self.model, "module") else self.model
        sigreg_loss, repulsion_loss = model.loss_adaptive_sigreg(
            sigreg_input, pseudo_label, mask, alpha, proj_x_lb, y_lb)

        return pred_loss, sigreg_loss, repulsion_loss, alpha

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, x_ulb_l, y_ulb, x_ulb_g=None):
        num_lb = y_lb.shape[0]

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs["logits"][:num_lb]
                proj_x_lb = outputs["proj"][:num_lb]
                proj_weak, proj_strong = outputs["proj"][num_lb:].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s = outputs["logits"][num_lb:].chunk(2)
                feats_x_lb = outputs["feat"][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs["feat"][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb["logits"]
                feats_x_lb = outs_x_lb["feat"]
                proj_x_lb = outs_x_lb["proj"]

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s["logits"]
                feats_x_ulb_s = outs_x_ulb_s["feat"]
                proj_strong = outs_x_ulb_s["proj"]

                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w["logits"]
                    feats_x_ulb_w = outs_x_ulb_w["feat"]
                    proj_weak = outs_x_ulb_w["proj"]

            feat_dict = {"x_lb": feats_x_lb, "x_ulb_w": feats_x_ulb_w, "x_ulb_s": feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction="mean")

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook(
                    "dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            mask = self.call_hook(
                "masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)

            pseudo_label = self.call_hook(
                "gen_ulb_targets", "PseudoLabelingHook",
                logits=probs_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T, softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, "ce", mask=mask)

            pred_loss, sigreg_loss, repulsion_loss, alpha = self._representation_loss(
                x_ulb_l, proj_weak, proj_strong, proj_x_lb, pseudo_label, mask, y_lb, x_ulb_g=x_ulb_g)

            rep_loss = (1.0 - self.beta) * pred_loss + self.beta * sigreg_loss + self.repulsion_coef * repulsion_loss
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_rep * rep_loss

            # Pseudo-labeling diagnostics (utilization / accuracy / class-count imbalance).
            mask_bool = mask.bool()
            num_pass = int(mask_bool.sum().item())
            num_ulb = mask.numel()
            pseudo_label_pass = pseudo_label[mask_bool]
            counts_pass = torch.bincount(pseudo_label_pass, minlength=self.num_classes)
            correct = int((pseudo_label_pass == y_ulb[mask_bool]).sum().item()) if num_pass > 0 else 0

            self.last_train_info = {
                "it": self.it,
                "num_ulb": num_ulb,
                "num_pass": num_pass,
                "util_ratio": num_pass / num_ulb if num_ulb > 0 else 0.0,
                "acc_pass": correct / num_pass if num_pass > 0 else 0.0,
                "max_count": int(counts_pass.max().item()),
                "min_count": int(counts_pass.min().item()),
            }

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            pred_loss=pred_loss.item(),
            sigreg_loss=sigreg_loss.item(),
            repulsion_loss=repulsion_loss.item(),
            rep_loss=rep_loss.item(),
            total_loss=total_loss.item(),
            alpha=float(alpha),
            acc_pass=self.last_train_info["acc_pass"],
            util_ratio=self.last_train_info["util_ratio"],
        )
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict["classwise_acc"] = self.hooks_dict["MaskingHook"].classwise_acc.cpu()
        save_dict["selected_label"] = self.hooks_dict["MaskingHook"].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict["MaskingHook"].classwise_acc = checkpoint["classwise_acc"].cuda(self.gpu)
        self.hooks_dict["MaskingHook"].selected_label = checkpoint["selected_label"].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--hard_label", str2bool, True),
            SSL_Argument("--T", float, 0.5),
            SSL_Argument("--p_cutoff", float, 0.95),
            SSL_Argument("--thresh_warmup", str2bool, True),
            SSL_Argument("--beta", float, 0.2, "SIGReg balance within the representation loss"),
            SSL_Argument("--lambda_rep", float, 0.5, "representation loss weight"),
            SSL_Argument("--warmup_ratio", float, 0.25, "fraction of training spent in the SIGReg warmup phase (paper's T_warm / T)"),
            SSL_Argument("--centroid_mean", str, "local",
                         "JEPA prediction-loss target: 'local' = mean of the K local crops; "
                         "'global' = mean of the global (weak + strong) views"),
            SSL_Argument("--sigreg_views", str, "local",
                         "views (Adaptive Class-wise) SIGReg regularizes: 'local' = local crops only; "
                         "'all' = weak + strong + local (the paper's 3 view types)"),
            SSL_Argument("--repulsion_coef", float, 1000.0,
                         "extra weight on the repulsion loss within the representation loss. "
                         "The paper's total-loss equation shows no explicit coefficient here (implicitly 1.0), "
                         "but the checkpoints behind every reported result were trained with this term scaled by "
                         "1000x -- the repulsion loss is otherwise numerically tiny relative to L_pred/L_sigreg "
                         "and has negligible effect on class-mean separation. Set to 1.0 to match the paper's "
                         "written equation exactly (weaker repulsion, less separated class means)."),
            SSL_Argument("--save_train_info", str2bool, False, "periodically dump per-step pseudo-labeling diagnostics (used for paper figures)"),
        ]
