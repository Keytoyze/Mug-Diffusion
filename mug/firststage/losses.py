from mug.model.models import *


class ManiaReconstructLoss(torch.nn.Module):

    def __init__(self, weight_start_offset=1.0, weight_holding=1.0, weight_end_offset=1.0,
                 label_smoothing=0.0, gamma=2.0):
        super(ManiaReconstructLoss, self).__init__()
        self.weight_start_offset = weight_start_offset
        self.weight_holding = weight_holding
        self.weight_end_offset = weight_end_offset
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def label_smoothing_bce_loss(self, predicts, targets):
        # p = torch.sigmoid(predicts)
        # p_t = p * targets + (1 - p) * (1 - targets)
        return self.bce_loss(
            predicts,
            targets * (1 - 2 * self.label_smoothing) + self.label_smoothing,
        )# * ((1 - p_t) ** self.gamma)

    def get_key_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, valid: torch.Tensor,
                     key_count, loss_func, index) -> torch.Tensor:
        loss = loss_func(
            reconstructions[:, index:index + key_count, :],
            inputs[:, index:index + key_count, :]
        )
        return torch.mean(loss * valid) / torch.mean(valid + 1e-6)

    def classification_metrics(self, inputs, reconstructions, valid_flag, key_count):
        predict_start = reconstructions >= 0
        true_start = inputs
        tp = true_start == predict_start
        tp_valid = tp * valid_flag
        acc_start = (torch.sum(tp_valid) /
                     (torch.sum(valid_flag) + 1e-5) / key_count
                     ).item()
        precision_start = (torch.sum(tp_valid * predict_start) /
                           (torch.sum(predict_start * valid_flag) + 1e-5)
                           ).item()
        recall_start = (torch.sum(tp_valid * true_start) /
                        (torch.sum(true_start * valid_flag) + 1e-5)
                        ).item()
        return acc_start, precision_start, recall_start

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                valid_flag: torch.Tensor):
        """
        inputs / reconstructions: [B, 4 * K, T]
        valid_flag: [B, T]
        Feature Layout:
            [is_start: 0/1] * key_count

            [offset_start: 0-1] * key_count
            valid only if is_start = 1

            [is_holding: 0/1] * key_count, (exclude start, include end),
            valid only if previous.is_start = 1 or previous.is_holding = 1

            [offset_end: 0-1]
            valid only if is_holding = 1 and latter.is_holding = 0
        """
        valid_flag = torch.ones_like(valid_flag) # TODO
        key_count = inputs.shape[1] // 4
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]
        T = inputs.shape[0]
        is_start = inputs[:, :key_count, :]  # [B, K, T]
        inputs_pad = torch.nn.functional.pad(inputs, (0, 1))  # [B, K, T + 1]
        is_end = (inputs[:, 2 * key_count:3 * key_count, :] -
                  inputs_pad[:, 2 * key_count: 3 * key_count, 1:] > 0.5).int()

        start_loss = self.get_key_loss(inputs, reconstructions, valid_flag,
                                       key_count,
                                       self.label_smoothing_bce_loss, 0)
        holding_loss = self.get_key_loss(inputs, reconstructions, valid_flag,
                                         key_count,
                                         self.label_smoothing_bce_loss, key_count * 2)
        offset_start_loss = self.get_key_loss(inputs, reconstructions,
                                              valid_flag * is_start,
                                              key_count,
                                              self.mse_loss, key_count)
        offset_end_loss = self.get_key_loss(inputs, reconstructions,
                                            valid_flag * is_end,
                                            key_count,
                                            self.mse_loss, key_count * 3)

        acc_start, precision_start, recall_start = self.classification_metrics(
            is_start, reconstructions[:, :key_count, :], valid_flag, key_count
        )
        acc_ln_start, precision_ln_start, recall_ln_start = self.classification_metrics(
            inputs[:, 2 * key_count:3 * key_count, :],
            reconstructions[:, 2 * key_count:3 * key_count, :],
            valid_flag, key_count
        )

        loss = (start_loss +
                holding_loss * self.weight_holding +
                offset_start_loss * self.weight_start_offset +
                offset_end_loss * self.weight_end_offset)
        return loss, {
            'start_loss': start_loss.detach().item(),
            'holding_loss': holding_loss.detach().item(),
            'offset_start_loss': offset_start_loss.detach().item(),
            'offset_end_loss': offset_end_loss.detach().item(),
            "acc_rice": acc_start,
            "acc_ln": acc_ln_start,
            "precision_rice": precision_start,
            "precision_ln": precision_ln_start,
            "recall_rice": recall_start,
            "recall_ln": recall_ln_start,
        }

class ManiaRhythmReconstructLoss(torch.nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(ManiaRhythmReconstructLoss, self).__init__()
        self.weight_start_offset = weight_start_offset
        self.weight_holding = weight_holding
        self.weight_end_offset = weight_end_offset
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.label_smoothing = label_smoothing

    def label_smoothing_bce_loss(self, predicts, targets):
        return self.bce_loss(
            predicts,
            targets * (1 - 2 * self.label_smoothing) + self.label_smoothing,
        )

    def get_key_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, valid: torch.Tensor,
                     key_count, loss_func, index) -> torch.Tensor:
        loss = loss_func(
            reconstructions[:, index:index + key_count, :],
            inputs[:, index:index + key_count, :]
        )
        return torch.mean(loss * valid) / torch.mean(valid + 1e-6)

    def classification_metrics(self, inputs, reconstructions, valid_flag, key_count):
        predict_start = reconstructions >= 0
        true_start = inputs
        tp = true_start == predict_start
        tp_valid = tp * valid_flag
        acc_start = (torch.sum(tp_valid) /
                     (torch.sum(valid_flag) + 1e-5) / key_count
                     ).item()
        precision_start = (torch.sum(tp_valid * predict_start) /
                           (torch.sum(predict_start * valid_flag) + 1e-5)
                           ).item()
        recall_start = (torch.sum(tp_valid * true_start) /
                        (torch.sum(true_start * valid_flag) + 1e-5)
                        ).item()
        return acc_start, precision_start, recall_start

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                valid_flag: torch.Tensor):
        """
        inputs / reconstructions: [B, 4 * K, T]
        valid_flag: [B, T]
        Feature Layout:
            [is_start: 0/1] * key_count

            [offset_start: 0-1] * key_count
            valid only if is_start = 1

            [is_holding: 0/1] * key_count, (exclude start, include end),
            valid only if previous.is_start = 1 or previous.is_holding = 1

            [offset_end: 0-1]
            valid only if is_holding = 1 and latter.is_holding = 0
        """
        valid_flag = torch.ones_like(valid_flag) # TODO
        key_count = inputs.shape[1] // 4
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]
        T = inputs.shape[0]
        is_start = inputs[:, :key_count, :]  # [B, K, T]
        inputs_pad = torch.nn.functional.pad(inputs, (0, 1))  # [B, K, T + 1]
        is_end = (inputs[:, 2 * key_count:3 * key_count, :] -
                  inputs_pad[:, 2 * key_count: 3 * key_count, 1:] > 0.5).int()

        start_loss = self.get_key_loss(inputs, reconstructions, valid_flag,
                                       key_count,
                                       self.label_smoothing_bce_loss, 0)
        holding_loss = self.get_key_loss(inputs, reconstructions, valid_flag,
                                         key_count,
                                         self.label_smoothing_bce_loss, key_count * 2)
        offset_start_loss = self.get_key_loss(inputs, reconstructions,
                                              valid_flag * is_start,
                                              key_count,
                                              self.mse_loss, key_count)
        offset_end_loss = self.get_key_loss(inputs, reconstructions,
                                            valid_flag * is_end,
                                            key_count,
                                            self.mse_loss, key_count * 3)

        acc_start, precision_start, recall_start = self.classification_metrics(
            is_start, reconstructions[:, :key_count, :], valid_flag, key_count
        )
        acc_ln_start, precision_ln_start, recall_ln_start = self.classification_metrics(
            inputs[:, 2 * key_count:3 * key_count, :],
            reconstructions[:, 2 * key_count:3 * key_count, :],
            valid_flag, key_count
        )

        loss = (start_loss +
                holding_loss * self.weight_holding +
                offset_start_loss * self.weight_start_offset +
                offset_end_loss * self.weight_end_offset)
        return loss, {
            'start_loss': start_loss.detach().item(),
            'holding_loss': holding_loss.detach().item(),
            'offset_start_loss': offset_start_loss.detach().item(),
            'offset_end_loss': offset_end_loss.detach().item(),
            "acc_rice": acc_start,
            "acc_ln": acc_ln_start,
            "precision_rice": precision_start,
            "precision_ln": precision_ln_start,
            "recall_rice": recall_start,
            "recall_ln": recall_ln_start,
        }