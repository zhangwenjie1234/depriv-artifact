import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
from utils.datasets import feature_sizes
from utils.criteo_preprocessing import CRITEO_NUMERIC_MASK
import torch.nn.functional as F
import torchvision
 
 
def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Flatten(nn.Module):
    '''Flatten the input'''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CIFARReleaseHead(nn.Module):
    '''
    Compress a high-dimensional CIFAR feature map into a low-dimensional
    release vector before cross-party sharing / defense.
    '''
    def __init__(self, in_channels, release_dim):
        super(CIFARReleaseHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(in_channels, release_dim),
            nn.BatchNorm1d(release_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class CIFARPassiveWithRelease(nn.Module):
    '''
    Wrap a CIFAR passive backbone with a low-dimensional release head so the
    shared representation is a compact vector instead of a raw feature map.
    '''
    def __init__(self, backbone, release_dim, in_channels=512):
        super(CIFARPassiveWithRelease, self).__init__()
        self.backbone = backbone
        self.release_head = CIFARReleaseHead(in_channels, release_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.release_head(x)

class Conv2_passive(nn.Module):
    '''
    Conv2 passive model for CIFAR-10 (Weakest Role).
    Shallow CNN.
    '''
    def __init__(self, num_passive):
        super(Conv2_passive, self).__init__()
        # 2 layers only
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )               
    def forward(self, x):
        emb = self.embeddings(x)
        return emb

class Conv2_passive_het(nn.Module):
    '''
    [FIX] Heterogeneous version for CIFAR-10.
    Aligns output channels (512) and spatial dimensions with ResNet.
    '''
    def __init__(self):
        super(Conv2_passive_het, self).__init__()
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample 1 (32->16)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample 2 (16->8)
            
            # [Added] Additional pooling to match ResNet's 3rd downsampling step (8->4)
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # [Added] Projection from 32 to 512 channels
            nn.Conv2d(32, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )               
    def forward(self, x):
        return self.embeddings(x)

class Conv3_passive_het(nn.Module):
    '''
    Heterogeneous mid-strength CNN for CIFAR-10.
    Sits between Conv2 and Conv4 while keeping the output interface aligned
    with the 512-channel release head expected by CIFARPassiveWithRelease.
    '''
    def __init__(self):
        super(Conv3_passive_het, self).__init__()
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4

            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.embeddings(x)

class Conv4_passive(nn.Module):
    '''
    Conv4 passive model for CIFAR-10/100.
    '''
    def __init__(self, num_passive):
        super(Conv4_passive, self).__init__()
        self.num_passive = num_passive
        

        if num_passive not in [1, 2, 4, 8]:
            raise ValueError("The number of passive parties must be 1, 2, 4 or 8.")
        
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )        
       
    def forward(self, x):
        emb = self.embeddings(x)
        return emb
    
class Conv4_passive_het(nn.Module):
    '''
    [FIX] Heterogeneous version for CIFAR-10.
    Aligns output channels (512) and spatial dimensions with ResNet.
    '''
    def __init__(self):
        super(Conv4_passive_het, self).__init__()
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample 1
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample 2
            nn.BatchNorm2d(64),

            # [Added] Additional pooling
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample 3

            # [Added] Projection to 512 channels
            nn.Conv2d(64, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )        
       
    def forward(self, x):
        return self.embeddings(x)

class Conv4_active(nn.Module):
    '''
    Conv4 active model for CIFAR10/100.
    '''
    def __init__(self, num_classes, num_passive, division_mode):
        super(Conv4_active, self).__init__()
        self.num_passive = num_passive

        if division_mode == 'imbalanced' and num_passive == 4:
            input_size = 8 * 7 * 64
        else:
            input_size = 8 * 8 * 64

        self.prediction = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        print("Active Model", self.prediction)
        
    def forward(self, x):
        logit = self.prediction(x)
        return logit

    
class Conv4(nn.Module):
    '''
    Conv4 model for CIFAR10/100.
    '''
    def __init__(self, num_classes, num_passive, padding_mode, division_mode, args=None):
        super(Conv4, self).__init__()
        self.num_passive = num_passive
        self.padding_mode = padding_mode

        use_hg = args is not None and args.hg
        if use_hg:
            raise NotImplementedError("--hg is only implemented for FC1 (mnist, fashionmnist) models.")

        self.passive = nn.ModuleList()
        for _ in range(num_passive):
            self.passive.append(Conv4_passive(num_passive))

        self.active = Conv4_active(num_classes, num_passive, division_mode)
        fusion_dropout = 0.0 if args is None else float(getattr(args, 'fusion_dropout', 0.0))
        self.fusion_dropout = nn.Dropout2d(p=fusion_dropout) if fusion_dropout > 0.0 else nn.Identity()
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))

        if self.padding_mode:
            agg_emb = self._aggregate_padding_mode(emb)  # agg_emb torch.Size([128, 64, 7, 7])
        else:
            agg_emb = self._aggregate(emb)
        agg_emb = self.fusion_dropout(agg_emb)
        logit = self.active(agg_emb)

        pred = self.softmax(logit)

        return emb, logit, pred

    def forward_from_embeddings(self, emb):
        if self.padding_mode:
            agg_emb = self._aggregate_padding_mode(list(emb))
        else:
            agg_emb = self._aggregate(list(emb))
        agg_emb = self.fusion_dropout(agg_emb)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)
        return logit, pred
    
    def _aggregate(self, x):
        '''
        Aggregate the embeddings from passive parties.
        '''
        return torch.cat(x, dim=3)
    
    def _aggregate_padding_mode(self, x):
        '''
        Aggregate the embeddings from passive parties using padding mode.
        '''
        return torch.sum(torch.stack(x, dim=0), dim=0)

    
class Flatten_input(nn.Module):
    '''Flatten the input'''
    def __init__(self):
        super(Flatten_input, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)
    

class FC1_passive(nn.Module):
    '''
    FC1 passive model for MNIST and FashionMNIST.
    28 * 28 * 1
    '''
    def __init__(self, num_passive, padding_mode, linear_size):
        super(FC1_passive, self).__init__()

        if num_passive not in [1, 2, 4, 7]:
            raise ValueError("The number of passive parties must be 1, 2, 4 or 7.")

        if padding_mode:
            self.embeddings = nn.Sequential(
                Flatten_input(),
                nn.Linear(28 * 28, 28 * 28),
                nn.BatchNorm1d(28 * 28),
                nn.ReLU()
            )
        else:
            emb_size = int(28 * (28 / num_passive))
            self.embeddings = nn.Sequential(
                Flatten_input(),
                nn.Linear(linear_size, emb_size),
                nn.BatchNorm1d(emb_size),
                nn.ReLU()
            )

       
    def forward(self, x):
        emb = self.embeddings(x)
        return emb

    
class FC1_active(nn.Module):
    '''
    Active model for MNIST and FashionMNIST.
    28 * 28 * 1
    '''
    def __init__(self, num_classes=10):
        super(FC1_active, self).__init__()

        self.prediction = nn.Sequential(
            nn.Linear(28 * 28, num_classes)
        )
        print("Active Model", self.prediction)
        
    def forward(self, x):
        logit = self.prediction(x)
        return logit

class MLP_passive_het(nn.Module):
    '''HG Exp: MLP 被动方模型 (Medium Role)'''
    def __init__(self, input_size, output_size):
        super(MLP_passive_het, self).__init__()
        self.embeddings = nn.Sequential(
            Flatten_input(), 
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        emb = self.embeddings(x)
        return emb

class CNN_passive_het_v2(nn.Module):
    '''HG Exp: Shallow CNN (Strong Role)'''
    def __init__(self, input_channels, output_size, input_width=7):
        super(CNN_passive_het_v2, self).__init__()
        dummy_input_w = input_width
        
        conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), 
        )
        
        with torch.no_grad():
            dummy_out = conv_layers(torch.randn(1, input_channels, 28, dummy_input_w))
            flattened_size = dummy_out.view(1, -1).size(1) 

        fc_layers = nn.Sequential(
            Flatten(),
            nn.Linear(flattened_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )
        
        self.embeddings = nn.Sequential(
            conv_layers,
            fc_layers
        )
    
    def forward(self, x):
        return self.embeddings(x)

class Conv4_passive_mnist(nn.Module):
    '''
    HG Exp: Deep CNN for MNIST (Strongest Role)
    4-Layer CNN
    '''
    def __init__(self, input_channels, output_size):
        super(Conv4_passive_mnist, self).__init__()
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Conv 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x -> 14x
            # Conv 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Conv 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x -> 7x
        )
        
        # 动态计算展平后的维度
        with torch.no_grad():
            dummy_w = max(4, 28 // 4) 
            dummy_input = torch.zeros(1, input_channels, 28, dummy_w)
            dummy_out = self.features(dummy_input)
            flattened_size = dummy_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
class FC1(nn.Module):
    '''
    Conv4 model for MNIST and FashionMNIST.
    28 * 28 * 1
    '''
    def __init__(self, num_passive, padding_mode, division_mode, args=None):
        super(FC1, self).__init__()
        self.num_passive = num_passive
        self.padding_mode = padding_mode

        use_hg = args is not None and args.hg
        
        if use_hg and padding_mode:
            raise ValueError("--hg cannot be used with --padding_mode.")

        if padding_mode:
            linear_size_list = [28 * 28] * num_passive
        else:
            linear_size_list = []
            if division_mode in ['vertical', 'random']:
                particle_size = int(28 * (28 / num_passive))
                linear_size_list = [particle_size] * num_passive
            elif division_mode == 'imbalanced':
                if num_passive == 1:
                    linear_size_list = [28 * 28]
                elif num_passive == 2:
                    linear_size_list.append(28 * 20)
                    linear_size_list.append(28 * 8)
                elif num_passive == 4:
                    linear_size_list.append(28 * 12)
                    linear_size_list.append(28 * 6)
                    linear_size_list.append(28 * 3)
                    linear_size_list.append(28 * 7)

        if padding_mode:
            output_emb_size_list = [28 * 28] * num_passive
        else:
            # 确保所有被动方的输出嵌入大小一致，以便 Active Party 拼接
            particle_size = int(28 * (28 / num_passive))
            output_emb_size_list = [particle_size] * num_passive

        self.passive = nn.ModuleList()
        if use_hg:
            # 这是您的新实验逻辑
            print(f"HG Exp: Loading heterogeneous models for num_passive={num_passive} on {args.dataset}")
            input_channels = 1 # for mnist/fashionmnist
            
            if num_passive == 1:
                print("HG Exp: Using 1 party (FC1)")
                self.passive.append(FC1_passive(num_passive, padding_mode, linear_size_list[0]))
            
            elif num_passive == 2:
                # 您的要求: 2个被动方 -> CNN 和 FC1
                print("HG Exp: Using 2 parties (ID 0: CNN, ID 1: FC1)")
                input_w = 28 // num_passive
                self.passive.append(CNN_passive_het_v2(input_channels, output_emb_size_list[0], input_width=input_w))
                self.passive.append(FC1_passive(num_passive, padding_mode, linear_size_list[1]))
            
            elif num_passive == 4:
                print("HG Exp: Using 4 parties (ID 0: MLP, ID 1: FC1, ID 2: resnet18, ID 3: CNN)") 
                self.passive.append(MLP_passive_het(linear_size_list[0], output_emb_size_list[0]))
                self.passive.append(FC1_passive(num_passive, padding_mode, linear_size_list[1]))
                self.passive.append(ResNet18_passive_het(input_channels, output_emb_size_list[2]))
                input_w = 28 // num_passive
                self.passive.append(CNN_passive_het_v2(input_channels, output_emb_size_list[3], input_width=input_w))

        else:
            for i in range(num_passive):
                self.passive.append(FC1_passive(num_passive, padding_mode, linear_size_list[i]))
        

        self.active = FC1_active()
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))

        if self.padding_mode:
            agg_emb = self._aggregate_padding_mode(emb)
        else:
            agg_emb = self._aggregate(emb)
        logit = self.active(agg_emb)

        pred = self.softmax(logit)

        return emb, logit, pred

    def forward_from_embeddings(self, emb):
        if self.padding_mode:
            agg_emb = self._aggregate_padding_mode(list(emb))
        else:
            agg_emb = self._aggregate(list(emb))
        logit = self.active(agg_emb)
        pred = self.softmax(logit)
        return logit, pred
    
    def _aggregate(self, x):
        '''
        Aggregate the embeddings from passive parties.
        '''
        # Note: x is a list of tensors.
        return torch.cat(x, dim=1)
    
    def _aggregate_padding_mode(self, x):
        '''
        Aggregate the embeddings from passive parties using padding mode.
        '''
        return torch.sum(torch.stack(x, dim=0), dim=0)


class ResNet18_passive_het(nn.Module):
    '''HG Exp: ResNet18 被动方模型 (用于 num_passive=4)'''
    def __init__(self, input_channels, output_size):
        super(ResNet18_passive_het, self).__init__()
        self.model = torchvision.models.resnet18(weights=None)
        
        # 修改 conv1 以接受1通道输入
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除 maxpool，因为它对于 28x7 这样的切片太剧烈了
        self.model.maxpool = nn.Identity()
        
        # 替换全连接层
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False
    )


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.block_conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel,outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
 
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self,x):
        out1 = self.block_conv(x)
        out2 = self.shortcut(x)+out1
        out2 = F.relu(out2)
        return out2


class ResNet_passive(nn.Module):  
    def __init__(self, block, layers, num_passive):
        super(ResNet_passive, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            # (n-f+2*p)/s+1,n=28,n=32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), #64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0) #64
        )
 
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1) #64
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2) #32
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2) #16
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2) #8
        
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet_active(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_p=0.3):
        super(ResNet_active, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, num_passive, padding_mode, division_mode, args=None):
        super(ResNet, self).__init__()
        self.num_passive = num_passive
        self.padding_mode = padding_mode
        requested_release_dim = 0 if args is None else int(getattr(args, 'cifar_release_dim', 128))
        self.release_dim = requested_release_dim if requested_release_dim > 0 else 512

        use_hg = args is not None and args.hg
        self.passive = nn.ModuleList()
        if use_hg:
            if num_passive == 4:
                print(f"HG Exp: Using 4 parties for CIFAR-10")
                print("  ID 0: Conv3 (Weakest)")
                print("  ID 1: ResNet34 (Strongest)")
                print("  ID 2: ResNet18 (Strong)")
                print("  ID 3: Conv4 (Medium)")

                # ID 0: Conv3
                self.passive.append(CIFARPassiveWithRelease(Conv3_passive_het(), self.release_dim))
                # ID 1: ResNet34
                self.passive.append(CIFARPassiveWithRelease(
                    ResNet_passive(ResidualBlock, [3, 4, 6, 3], num_passive),
                    self.release_dim,
                ))
                # ID 2: ResNet18
                self.passive.append(CIFARPassiveWithRelease(
                    ResNet_passive(ResidualBlock, [2, 2, 2, 2], num_passive),
                    self.release_dim,
                ))
                # ID 3: Conv4
                self.passive.append(CIFARPassiveWithRelease(Conv4_passive_het(), self.release_dim))

            elif num_passive == 2:
                print(f"HG Exp: Using 2 parties for CIFAR-10")
                print("  ID 0: Conv4 (Medium)")
                print("  ID 1: ResNet18 (Strong)")
                # ID 0: Conv4
                self.passive.append(CIFARPassiveWithRelease(Conv4_passive_het(), self.release_dim))
                # ID 1: ResNet18
                self.passive.append(CIFARPassiveWithRelease(
                    ResNet_passive(ResidualBlock, [2, 2, 2, 2], num_passive),
                    self.release_dim,
                ))

        else:
            for i in range(num_passive):
                self.passive.append(CIFARPassiveWithRelease(
                    ResNet_passive(block, layers, num_passive),
                    self.release_dim,
                ))

        fusion_dropout = 0.3 if args is None else float(getattr(args, 'fusion_dropout', 0.3))
        self.fusion_dropout = nn.Dropout(p=fusion_dropout) if fusion_dropout > 0.0 else nn.Identity()
        dropout_p = fusion_dropout
        self.active = ResNet_active(self.release_dim * self.num_passive, num_classes, dropout_p=dropout_p)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))
        agg_emb = torch.cat(emb, dim=1)
        agg_emb = self.fusion_dropout(agg_emb)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)

        return emb, logit, pred

    def forward_from_embeddings(self, emb):
        agg_emb = torch.cat(list(emb), dim=1)
        agg_emb = self.fusion_dropout(agg_emb)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)
        return logit, pred
    
    def _aggregate(self, x):
        agg_emb = torch.cat(x, dim=1)
        agg_emb = agg_emb.view(agg_emb.size(0), -1)
        return agg_emb
    
    def _aggregate_padding_mode(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)


class CriteoResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super(CriteoResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class DeepFM_passive(nn.Module):
    '''
    DeepFM for criteo.
    '''
    def __init__(
        self,
        feature_sizes,
        emb_size,
        hidden_size,
        dropout,
        numeric_mask,
        backbone="mlp",
    ):
        super(DeepFM_passive, self).__init__()
        if len(feature_sizes) != len(numeric_mask):
            raise ValueError("Criteo feature sizes and numeric mask must align.")
        self.numeric_mask = tuple(bool(value) for value in numeric_mask)
        self.embeddings = nn.ModuleList([nn.Embedding(fz, emb_size) for fz in feature_sizes])
        input_width = len(feature_sizes) * emb_size
        if backbone == "linear":
            self.DNN = nn.Sequential(
                nn.Linear(input_width, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )
        elif backbone == "mlp":
            self.DNN = nn.Sequential(
                nn.Linear(input_width, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
            )
        elif backbone == "residual_mlp":
            residual_width = hidden_size * 2
            self.DNN = nn.Sequential(
                nn.Linear(input_width, residual_width),
                nn.BatchNorm1d(residual_width),
                nn.ReLU(),
                nn.Dropout(dropout),
                CriteoResidualBlock(residual_width, dropout),
                CriteoResidualBlock(residual_width, dropout),
                nn.Linear(residual_width, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )
        else:
            raise ValueError("Unsupported Criteo passive backbone: {}".format(backbone))
        self.backbone = backbone
        print(
            "Passive Model -> Criteo backbone={}, fields={}, release_dim={}".format(
                backbone, len(feature_sizes), hidden_size
            )
        )

    def forward(self, x):
        if x.ndim != 2 or x.size(1) != len(self.embeddings):
            raise ValueError(
                "Criteo party input width {} does not match its {} fields.".format(
                    x.size(1) if x.ndim >= 2 else None, len(self.embeddings)
                )
            )

        emb_fm = []
        for index, (embedding, is_numeric) in enumerate(
            zip(self.embeddings, self.numeric_mask)
        ):
            if is_numeric:
                indices = torch.zeros_like(x[:, index], dtype=torch.long)
                values = x[:, index].float()
            else:
                indices = x[:, index].long()
                if torch.any(indices < 0) or torch.any(indices >= embedding.num_embeddings):
                    raise ValueError(
                        f"Criteo categorical field {index} has an out-of-range bucket."
                    )
                values = torch.ones_like(x[:, index], dtype=torch.float32)
            emb_fm.append(embedding(indices) * values.unsqueeze(1))

        # Deep part
        emb_deep = torch.cat(emb_fm, dim=1)
        emb_deep = self.DNN(emb_deep)

        return emb_deep


class DeepFM_active(nn.Module):
    def __init__(self, hidden_size, dropout, num_classes, num_passive):
        super(DeepFM_active, self).__init__()
        if num_passive == 1:
            self.prediction = nn.Sequential(
                nn.Linear(hidden_size, num_classes)
            )
        elif num_passive == 3:
            self.prediction = nn.Sequential(
                nn.Linear(hidden_size * num_passive, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            raise ValueError("The number of passive parties must be 1 or 3.")
        print("Active Model", self.prediction)

    def forward(self, x):
        logit = self.prediction(x)
        return logit


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, emb_size, hidden_size, dropout, num_classes, num_passive, padding_mode, division_mode, args=None):
        super(DeepFM, self).__init__()
        self.num_passive = num_passive
        use_hg = bool(args is not None and getattr(args, "hg", False))

        if len(feature_sizes) != len(CRITEO_NUMERIC_MASK):
            raise ValueError(
                "Criteo preprocessing must initialize all 39 feature sizes before "
                "constructing the model."
            )
        if len(feature_sizes) % num_passive != 0:
            raise ValueError("Criteo features must divide evenly across parties.")
        if use_hg and num_passive != 3:
            raise ValueError("Criteo --hg requires --num_passive 3.")

        feature_size_list = []
        numeric_mask_list = []
        feature_stride = int(len(feature_sizes) / num_passive)
        for i in range(num_passive):
            feature_size_list.append(feature_sizes[i*feature_stride: (i+1)*feature_stride])
            numeric_mask_list.append(
                CRITEO_NUMERIC_MASK[i*feature_stride: (i+1)*feature_stride]
            )

        self.passive = nn.ModuleList()
        backbones = ["linear", "mlp", "residual_mlp"] if use_hg else ["mlp"] * num_passive
        if use_hg:
            print(
                "HG Exp: Criteo parties -> P0 Linear, P1 MLP, "
                "P2 ResidualMLP (shared release_dim={})".format(hidden_size)
            )
        for i in range(num_passive):
            self.passive.append(
                DeepFM_passive(
                    feature_size_list[i],
                    emb_size,
                    hidden_size,
                    dropout,
                    numeric_mask_list[i],
                    backbone=backbones[i],
                )
            )

        self.active = DeepFM_active(hidden_size, dropout, num_classes, num_passive)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))

        logit, pred = self.forward_from_embeddings(emb)

        return emb, logit, pred

    def forward_from_embeddings(self, emb):
        """Continue at the active party from released party embeddings.

        ``defense_all``, PPDL, ACVFL, and AdaVFed all perturb or detach the
        VFL boundary tensors before active-side inference.  Keeping that path
        here also guarantees that clean and defended DeepFM runs use the same
        aggregation and prediction logic.
        """
        embeddings = list(emb)
        if len(embeddings) != self.num_passive:
            raise ValueError(
                "Criteo DeepFM expected {} party embeddings, got {}.".format(
                    self.num_passive, len(embeddings)
                )
            )
        if not embeddings:
            raise ValueError("Criteo DeepFM requires at least one party embedding.")
        for party_id, embedding in enumerate(embeddings):
            if embedding.ndim != 2:
                raise ValueError(
                    "Criteo party {} embedding must be two-dimensional, got {}.".format(
                        party_id, tuple(embedding.shape)
                    )
                )
        batch_size = embeddings[0].shape[0]
        for embedding in embeddings:
            if embedding.shape[0] != batch_size:
                raise ValueError("Criteo party embeddings have different batch sizes.")

        agg_emb = self._aggregate(embeddings)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)
        return logit, pred
    
    def _aggregate(self, x):
        return torch.cat(x, dim=1)



entire = {
    'mnist': FC1,
    'fashionmnist': FC1,
    'cifar10': partial(ResNet, block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=10),
    'cifar100': partial(ResNet, block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=100),
    'criteo': partial(DeepFM, feature_sizes=feature_sizes, emb_size=4, hidden_size=32, dropout=0.5, num_classes=2)
}

entire_simple = {
    'mnist': FC1,
    'fashionmnist': FC1,
    'cifar10': partial(Conv4, num_classes=10),
    'cifar100': partial(ResNet, block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=100),
    'criteo': partial(DeepFM, feature_sizes=feature_sizes, emb_size=4, hidden_size=32, dropout=0.5, num_classes=2)
}
