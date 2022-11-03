



import torch
import torch_geometric
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, LEConv, global_mean_pool


class GCN_Xplain_v8_ap(torch.nn.Module):
    def __init__(self,args):
        super(GCN_Xplain_v8_ap, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        # for explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        H_0, H_1, H_2, H_3 = self.num_features,128,256,512

        # GCN layers        
        self.conv1_le = LEConv(self.num_features, H_1)
        self.conv1 = GCNConv(H_1, H_1)
        
        self.conv2_le = LEConv(H_1, H_2)
        self.conv2 = GCNConv(H_2, H_2)

        self.conv3_le = LEConv(H_2, H_3)
        self.conv3 = GCNConv(H_3, H_3)
        self.lin1 = torch.nn.Linear(H_3, H_2)
        self.lin2 = torch.nn.Linear(H_2, H_1)
        self.lin3 = torch.nn.Linear(H_1, 1)

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h0.requires_grad = True
        self.input = h0
        h0 = F.relu(self.conv1_le(h0, edge_index, edge_weight))
        h1 = F.relu(self.conv1(h0, edge_index, edge_weight))
        h1 = F.relu(self.conv2_le(h1, edge_index, edge_weight))
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.relu(self.conv3_le(h2, edge_index, edge_weight))
        with torch.enable_grad():
            self.final_conv_acts = self.conv3(h2, edge_index, edge_weight)
        self.final_conv_acts.register_hook(self.activations_hook)
        h3 = F.relu(self.final_conv_acts)
        h4 = global_mean_pool(h3, data.batch)
        
        x = F.relu(self.lin1(h4))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        

        return x