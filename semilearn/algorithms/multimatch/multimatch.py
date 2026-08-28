import torch
import torch.nn.functional as F

from semilearn.algorithms.flexmatch.utils import FlexMatchThresholdingHook
from semilearn.algorithms.freematch.utils import FreeMatchThresholingHook as FreeMatchThresholdingHook
from semilearn.algorithms.multimatch.utils import MultiMatchThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('multimatch')
class MultiMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        
        # multihead specific arguments
        self.num_heads = args.num_heads

        # arguments used by the auxiliary thresholding (e.g. freematch)
        self.init_maskinghook_args(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile,
                                   clip_thresh=args.clip_thresh, p_cutoff=args.p_cutoff, thresh_warmup=args.thresh_warmup,
                                   threshold_algo=args.threshold_algo)

        super().__init__(args, net_builder, tb_log, logger) 


    def init_maskinghook_args(self, T, p_cutoff, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False, thresh_warmup=True, threshold_algo='freematch'):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh
        self.threshold_algo = threshold_algo

    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(self.args)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(self.args)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model
    
    def set_hooks(self):
        self.register_hook(MultiMatchThresholdingHook(self.args), "APMHook")

        for i in range(self.num_heads):
            if self.threshold_algo == 'flexmatch':
                self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), f"MaskingHook{i}")
            elif self.threshold_algo == 'freematch':
                self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), f"MaskingHook{i}")
            elif self.threshold_algo == 'none':
                pass
            else:
                raise NotImplementedError()

        super().set_hooks()

    def get_head_logits(self, head_id, logits, num_lb):
        head_logits = logits[head_id]
        logits_x_lb = head_logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = head_logits[num_lb:].chunk(2)
        return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s
    
    def get_pseudo_labels(self, ulb_weak_logits):
        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        _, pseudo_labels = torch.max(ulb_weak_logits, dim=-1)
        return pseudo_labels
    
    def get_supervised_loss(self, lb_logits, lb_target):
        head_losses = [F.cross_entropy(lb_logits[head_id], lb_target) for head_id in range(self.num_heads)]
        if self.args.average_losses:
            return sum(head_losses) / len(head_losses)
        return sum(head_losses)

    def _get_auxiliary_mask(self, logits_x_ulb_w, idx_ulb, head_id):
        # calculate mask
        if self.threshold_algo == 'freematch':
            mask = self.call_hook("masking", f"MaskingHook{head_id}", logits_x_ulb=logits_x_ulb_w)
        elif self.threshold_algo == 'flexmatch':
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            mask = self.call_hook("masking", f"MaskingHook{head_id}", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
        elif self.threshold_algo == 'none':
            mask = torch.ones(idx_ulb.shape[0], dtype=torch.int64).cuda(self.gpu)
        else:
            raise NotImplementedError()
        return mask
    
    def get_auxiliary_mask_comp(self, logits_x_ulb_w, idx_ulb, head_id1, head_id2):
        auxiliary_mask1 = self._get_auxiliary_mask(logits_x_ulb_w[head_id1], idx_ulb, head_id1)
        auxiliary_mask2 = self._get_auxiliary_mask(logits_x_ulb_w[head_id2], idx_ulb, head_id2)
        return torch.maximum(auxiliary_mask1, auxiliary_mask2)

    def get_head_unsupervised_loss(self, ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, y_ulb, head_id):
        '''
        This works only for 3 heads
        '''
        if head_id == 0:
            head_id1, head_id2 = 1, 2
        elif head_id == 1:
            head_id1, head_id2 = 0, 2
        else:
            head_id1, head_id2 = 0, 1

        num_ulb = idx_ulb.shape[0]
        multihead_labels = torch.ones(num_ulb, dtype=torch.int64).cuda(self.gpu) * -1
        multihead_agreement_types = torch.ones(num_ulb, dtype=torch.int64).cuda(self.gpu) * -1
        agreement_types_mask = torch.ones(num_ulb, dtype=torch.int64).cuda(self.gpu) * -1

        for i in range(num_ulb):
            label1 = pseudo_labels[head_id1][i]
            label2 = pseudo_labels[head_id2][i]
            multihead_labels[i], multihead_agreement_types[i], agreement_types_mask[i] = self.call_hook(
                "get_apm_label", "APMHook", head_id=head_id, head_id1=head_id1, head_id2=head_id2, idx=idx_ulb[i], label1=label1, label2=label2)
        
        auxiliary_mask = self.get_auxiliary_mask_comp(ulb_weak_logits, idx_ulb, head_id1, head_id2)

        multihead_labels[multihead_labels == -1] = 0 # can't have labels -1, even though the weight will be 0
        samples_weights = (agreement_types_mask == 0) * self.args.apm_disagreement_weight + (agreement_types_mask == 1) * 1

        final_weights = samples_weights * auxiliary_mask

        return (F.cross_entropy(ulb_strong_logits[head_id], multihead_labels, reduction='none') * final_weights).mean()


    def get_unsupervised_loss(self, ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, y_ulb):
        for head_id in range(self.num_heads):
            self.call_hook("update", "APMHook", logits_x_ulb_w=ulb_weak_logits[head_id], logits_x_ulb_s=ulb_strong_logits[head_id], idx_ulb=idx_ulb, head_id=head_id)
        
        head_losses = [self.get_head_unsupervised_loss(ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, y_ulb, head_id) for head_id in range(self.num_heads)]
        return sum(head_losses) / self.num_heads
    
    def get_loss(self, lb_loss, ulb_loss):
        return lb_loss + self.lambda_u * ulb_loss
    
    def _post_process_logits(self, logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, y_lb, idx_ulb, y_ulb, feat_dict=None):
         # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, y_lb)

        # Pseudo labels   
        pseudo_labels = torch.stack([self.get_pseudo_labels(logits_x_ulb_w[head_id]) for head_id in range(self.num_heads)])

        # Unsupervised loss
        ulb_loss = self.get_unsupervised_loss(logits_x_ulb_w, logits_x_ulb_s, pseudo_labels, idx_ulb, y_ulb)

        # Total loss
        loss = self.get_loss(lb_loss, ulb_loss)

        if feat_dict:
            out_dict = self.process_out_dict(loss=loss, feat=feat_dict)
        else:
            out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(sup_loss=lb_loss.item(), 
                                         unsup_loss=ulb_loss.item(), 
                                         total_loss=loss.item())
        
        return out_dict, log_dict

    def train_step_base(self, logits, y_lb, idx_ulb, y_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = idx_ulb.shape[0]

        logits_x_lb = torch.zeros(self.num_heads, num_lb, self.num_classes).cuda(self.gpu)
        logits_x_ulb_w = torch.zeros(self.num_heads, num_ulb, self.num_classes).cuda(self.gpu)
        logits_x_ulb_s = torch.zeros(self.num_heads, num_ulb, self.num_classes).cuda(self.gpu)

        for head_id in range(self.num_heads):
            logits_x_lb[head_id], logits_x_ulb_w[head_id], logits_x_ulb_s[head_id] = \
                self.get_head_logits(head_id, logits, num_lb)

        return self._post_process_logits(logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, y_lb, idx_ulb, y_ulb)


    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_ulb, y_ulb=None):       
        idx_ulb = idx_ulb.cuda(self.gpu)

        if self.use_cat:
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            inputs = inputs.cuda(self.gpu)
            logits = self.model(inputs)['logits']
            return self.train_step_base(logits, y_lb, idx_ulb, y_ulb)
        else:
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']
            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            return self._post_process_logits(logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, y_lb, idx_ulb, y_ulb, feat_dict=feat_dict)

    def get_logits(self, data, out_key):
        x = data['x_lb']
        if isinstance(x, dict):
            x = {k: v.cuda(self.gpu) for k, v in x.items()}
        else:
            x = x.cuda(self.gpu)
        
        logits = self.model(x)[out_key]

        # Use all heads for prediction
        return sum(logits) / self.num_heads

    def get_save_dict(self):
        save_dict = super().get_save_dict()

        # additional saving arguments
        for i in range(self.num_heads):
            if self.threshold_algo == 'freematch':
                save_dict[f'p_model{i}'] = self.hooks_dict[f'MaskingHook{i}'].p_model.cpu()
                save_dict[f'time_p{i}'] = self.hooks_dict[f'MaskingHook{i}'].time_p.cpu()
            elif self.threshold_algo == 'flexmatch':
                save_dict[f'classwise_acc{i}'] = self.hooks_dict[f'MaskingHook{i}'].classwise_acc.cpu()
                save_dict[f'selected_label{i}'] = self.hooks_dict[f'MaskingHook{i}'].selected_label.cpu()
            elif self.threshold_algo == 'none':
                pass
            else:
                raise NotImplementedError()

        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        
        for i in range(self.num_heads):
            if self.threshold_algo == 'freematch':
                self.hooks_dict[f'MaskingHook{i}'].p_model = checkpoint[f'p_model{i}'].cuda(self.gpu)
                self.hooks_dict[f'MaskingHook{i}'].time_p = checkpoint[f'time_p{i}'].cuda(self.gpu)
            elif self.threshold_algo == 'flexmatch':
                self.hooks_dict[f'MaskingHook{i}'].classwise_acc = checkpoint[f'classwise_acc{i}'].cuda(self.gpu)
                self.hooks_dict[f'MaskingHook{i}'].selected_label = checkpoint[f'selected_label{i}'].cuda(self.gpu)
            elif self.threshold_algo == 'none':
                pass
            else:
                raise NotImplementedError()
            
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--num_heads', int, 3),
            SSL_Argument('--no_low', str2bool, False), # gamma_min -inf (True) or 0 (False), the lower limit for the apm threshold
            SSL_Argument('--apm_disagreement_weight', float, 3),
            SSL_Argument('--apm_percentile', float, 0.05),
            SSL_Argument('--smoothness', float, 0.997),
            SSL_Argument('--adjust_clf_size', str2bool, False),
            SSL_Argument('--num_recalibrate_iter', int, 0), # if 0, it will be done every epoch
            SSL_Argument('--average_losses', str2bool, False),
            SSL_Argument('--threshold_algo', str, 'freematch'),
            # arguments used by the freematch/flexmatch thresholding
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
