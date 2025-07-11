import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimatorCV:
    def __init__(self, feature_num, class_num):
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        features_by_sort = features.view(N, 1, A) * labels.view(N, C, 1)

        Amount_CxA = labels.view(N, C, 1).expand(N, C, A).sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.view(1, C, A) * labels.view(N, C, 1)
        var_temp = (var_temp**2).sum(0) / Amount_CxA

        sum_weight_CV = labels.sum(0)
        weight_CV = sum_weight_CV / (sum_weight_CV + self.Amount)
        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = (weight_CV * (1 - weight_CV)).view(C, 1) * (self.Ave - ave_CxA) ** 2

        self.CoVariance = (
            self.CoVariance * (1 - weight_CV.view(C, 1)) + var_temp * weight_CV.view(C, 1)
        ).detach() + additional_CV.detach()
        self.Ave = (self.Ave * (1 - weight_CV.view(C, 1)) + ave_CxA * weight_CV.view(C, 1)).detach()
        self.Amount += labels.sum(0)


class ISDA_BCELoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDA_BCELoss, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num

    def isda_aug(self, fc, features, labels, cv_matrix, ratio):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        CV_temp = cv_matrix[labels]

        sigma2 = ((weight_m**2).view(1, C, A) * CV_temp.view(N, 1, A)).sum(2) * ratio * 0.5

        return sigma2

    def forward(self, model, output, features, target, ratio):
        self.estimator.update_CV(features.detach(), target)

        # 处理包装后的模型，提取原始模型
        original_model = model
        if hasattr(model, "model"):  # 如果是包装后的模型
            original_model = model.model
        elif hasattr(model, "module"):  # 如果是DataParallel或DistributedDataParallel包装的
            original_model = model.module
            if hasattr(original_model, "model"):  # 如果是双重包装
                original_model = original_model.model

        # 提取全连接层
        if hasattr(original_model, "fc"):
            fc = original_model.fc
        elif hasattr(original_model, "classifier"):
            if isinstance(original_model.classifier, nn.Sequential):
                fc = original_model.classifier[-1]
            else:
                fc = original_model.classifier
        else:
            raise AttributeError("Cannot find fc or classifier layer in the model")

        isda_aug_y = self.isda_aug(fc, features, target, self.estimator.CoVariance.detach(), ratio)
        augmented_output = output + isda_aug_y * (1 - 2 * target)

        loss = nn.BCEWithLogitsLoss()(augmented_output, target)

        return loss
