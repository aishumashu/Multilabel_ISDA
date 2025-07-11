import torch.nn as nn


class FeatureExtractor:
    """用于提取模型倒数第二层特征的类"""

    def __init__(self):
        self.features = None

    def hook(self, module, input, output):
        """Hook函数，用于捕获中间层输出"""
        self.features = output

    def get_features(self):
        return self.features


class ModelWithFeatures(nn.Module):
    """包装模型以同时返回最终输出和倒数第二层特征"""

    def __init__(self, model):
        super(ModelWithFeatures, self).__init__()
        self.model = model
        self.feature_extractor = FeatureExtractor()

        # 根据模型类型注册hook
        if hasattr(model, "fc"):
            # ResNet类型的模型
            if hasattr(model, "avgpool"):
                model.avgpool.register_forward_hook(self.feature_extractor.hook)
        elif hasattr(model, "classifier"):
            # VGG、AlexNet等类型的模型
            if isinstance(model.classifier, nn.Sequential):
                # 注册到classifier的倒数第二层
                model.classifier[-2].register_forward_hook(self.feature_extractor.hook)
            else:
                # 对于单层classifier，注册到features的最后一层
                if hasattr(model, "features"):
                    model.features[-1].register_forward_hook(self.feature_extractor.hook)

    def forward(self, x, isda=False):
        """
        前向传播
        Args:
            x: 输入张量
            isda: 是否返回倒数第二层特征
        Returns:
            如果isda=False: 只返回最终输出
            如果isda=True: 返回(最终输出, 倒数第二层特征)
        """
        output = self.model(x)
        if isda:
            features = self.feature_extractor.get_features()
            return output, features
        else:
            return output
