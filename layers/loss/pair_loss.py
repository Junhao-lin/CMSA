import torch
import torch.nn as nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    def forward(self, feature_maps, pair_labels):
        assert len(feature_maps) % 2 == 0, "Feature maps length should be even."
        
        # Apply global average pooling to change the shape to [batch_size, num_features]
        feature_maps = nn.functional.adaptive_avg_pool2d(feature_maps, (1, 1)).view(len(feature_maps), -1)
        
        similarities = []
        for i in range(0, len(feature_maps), 2):
            similarity = self.cosine_similarity(feature_maps[i].unsqueeze(0), feature_maps[i + 1].unsqueeze(0))
            similarities.append(1-similarity)

        similarities = torch.stack(similarities)
        loss = torch.sum(similarities).float()
        
        return loss