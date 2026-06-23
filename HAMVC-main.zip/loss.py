import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device

    def positive_contrastive_loss(self, z_view1, z_view2, lambda_uniformity=1.0, lambda_alignment=1.0):
        cos_sim = F.cosine_similarity(z_view1, z_view2, dim=1)
        uniformity_loss = (cos_sim ** 2).mean()
        mu1 = z_view1.mean(dim=0, keepdim=True)
        mu2 = z_view2.mean(dim=0, keepdim=True)
        z1_centered = z_view1 - mu1
        z2_centered = z_view2 - mu2

        alignment_loss = F.mse_loss(z1_centered, z2_centered)
        total_loss = (
            lambda_uniformity * uniformity_loss +
            lambda_alignment * alignment_loss
        )
        return total_loss
    
    


        