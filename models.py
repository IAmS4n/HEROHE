import math
from itertools import chain

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class PoolMultiRangeGM(nn.Module):
    def __init__(self, r_min, r_max, fs_num):
        super(PoolMultiRangeGM, self).__init__()

        print("[Pooling] Multi Generalized Mean Range(%.2f~%.2f)" % (r_min, r_max))
        d = float(r_max - r_min) / float(fs_num)

        r = torch.tensor([r_min + float(i) * d for i in range(fs_num)], requires_grad=False)
        self.register_buffer("r", r.view(1, 1, fs_num))

    def forward(self, x):
        # pos_bound_inp = torch.abs(x)  # negative number needs complex number
        # pos_bound_inp = torch.sigmoid(x)

        # alpha for numerical stability
        alpha = x.max(dim=1, keepdim=True)[0].detach()
        # alpha = torch.max(torch.ones_like(alpha), alpha).detach()

        normalize_inp = x / alpha
        normalize_output = (normalize_inp ** self.r).mean(1) ** (1. / self.r[:, 0, :])
        output = normalize_output * alpha[:, 0, :]
        return output


ModelProperty = {
    "efficientnet-b0":
        {"feature_num": 1280,
         "image_size": 224},

    "efficientnet-b1":
        {"feature_num": 1280,
         "image_size": 240},

    "efficientnet-b2":
        {"feature_num": 1408,
         "image_size": 260},

    "efficientnet-b3":
        {"feature_num": 1536,
         "image_size": 300},

    "efficientnet-b4":
        {"feature_num": 1792,
         "image_size": 380},

    "efficientnet-b5":
        {"feature_num": 2048,
         "image_size": 456},

    "efficientnet-b6":
        {"feature_num": 2304,
         "image_size": 528},

    "efficientnet-b7":
        {"feature_num": 2560,
         "image_size": 600},

    "inceptionresnetv2":
        {"feature_num": 1536,
         "image_size": 299}
}


class RealMaxPool1d(nn.Module):
    def __init__(self, kernel_size, padding):
        super(RealMaxPool1d, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.pooling(x[:, None, :])[:, 0, :]


def make_end_part(feature_num, out_num):
    # There is a dropout layer in main code before FC
    # Input is a 1D vector

    # return nn.Sequential(nn.Dropout(p=0.5, inplace=True),
    #                      nn.Linear(feature_num, out_num),
    #                      nn.BatchNorm1d(out_num, affine=False)
    #                      )
    assert feature_num > out_num
    kernel_size = math.ceil(feature_num / out_num)
    out_num_diff = out_num - math.floor(feature_num / kernel_size)
    assert out_num_diff >= 0

    # dirty
    padding_size = 0
    while out_num_diff > 0:
        padding_size += 1
        out_num_diff = out_num - math.floor((feature_num + 2 * padding_size) / kernel_size)
    assert out_num_diff == 0

    return nn.Sequential(  # nn.Dropout(p=0.25, inplace=True),
        RealMaxPool1d(kernel_size=kernel_size, padding=padding_size),
        nn.BatchNorm1d(out_num, affine=False)  # nn.Sigmoid()
    )


def make_pool(pool_name, fs_num):
    if pool_name.startswith("RGM"):
        conf = pool_name[3:]
        r_min, r_max = list(map(float, conf.split("-")))
        return PoolMultiRangeGM(r_min=r_min, r_max=r_max, fs_num=fs_num)
    else:
        raise ValueError("Invalide pooling type")


def make_classifier_head(fs_num, class_num, drop_out):
    ms = class_num
    if type(drop_out) in [int, float]:
        return nn.Sequential(
            nn.Dropout(p=drop_out, inplace=True),
            nn.Linear(fs_num, ms),
        )
    else:
        return nn.Sequential(
            nn.Linear(fs_num, ms),

        )


class Model(nn.Module):
    def __init__(self, name, out_num, heads, heads_pool, drop_outs):
        super(Model, self).__init__()
        self.name = name
        self.out_num = out_num

        assert len(heads) == len(heads_pool)

        self.heads = nn.ModuleList()
        for i, (head, drop_out) in enumerate(zip(heads, drop_outs)):
            self.heads.append(make_classifier_head(fs_num=out_num, class_num=head, drop_out=drop_out))

        self.heads_pool = nn.ModuleList()
        for head_pool in heads_pool:
            self.heads_pool.append(make_pool(head_pool, out_num))

        if name.startswith("efficientnet-b") and name in ModelProperty:
            feature_num = ModelProperty[name]["feature_num"]
            self.image_size = ModelProperty[name]["image_size"]

            self.main_model = EfficientNet.from_pretrained(name)
            self.main_model._fc = make_end_part(feature_num, out_num)
            self.efficientnet_family = True
        elif name.startswith("modified_efficientnet-b"):
            effnet_name = name.split("_")[1]
            feature_num = ModelProperty[effnet_name]["feature_num"]
            self.image_size = int(name.split("_")[2])

            self.main_model = EfficientNet.from_pretrained(effnet_name)
            self.main_model._fc = make_end_part(feature_num, out_num)
            self.efficientnet_family = True
        else:
            raise ValueError("Name not listed...")

        # freez_efficientnet(self.main_model, 1,freeze_last_conv=True)
        # freez_efficientnet(self.main_model, 0.5)

    def forward(self, input, return_feature=False, head_num=0, return_pooled_feature=False):
        assert not (return_feature and return_pooled_feature)

        assert len(input.size()) == 5 and input.size(2) == 3, input.size()  # batch path c w h
        batch_num, patch_num = list(input.size())[:2]
        img_shape = list(input.size())[2:]

        input = input.view([batch_num * patch_num] + img_shape)

        outputs = self.main_model(input)

        outputs = torch.abs(outputs)  # for compatibility with GM

        assert len(outputs.size()) == 2
        outputs = outputs.view([batch_num, patch_num, self.out_num])

        if return_feature:
            return outputs
        # outputs = backward_devide_patch_num(outputs)
        outputs = self.heads_pool[head_num](outputs)

        if return_pooled_feature:
            return outputs

        logit = self.heads[head_num](outputs)
        return logit

    def load_body_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith("main_model")}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict, strict=False)

    def parameters_model(self):
        return chain(self.main_model.parameters(), self.heads_pool.parameters(), self.heads.parameters())

    def parameters_model_linear_lr(self, lower, upper, group=4):
        assert self.efficientnet_family
        res = []

        # for layer in [model.main_model._conv_stem, model.main_model._bn0]:
        #     res.append({"params": layer.parameters(), "lr": lower})

        # Linear:
        # d = float(upper - lower) / float(len(model.main_model._blocks))
        # clr = lower
        # for inx, layer in enumerate(model.main_model._blocks):
        #     res.append({"params": layer.parameters(), "lr": clr})
        #     clr += d

        # Exponential:
        # d = (upper/lower) ** (1./float(len(model.main_model._blocks)))
        # clr = lower
        # for inx, layer in enumerate(model.main_model._blocks):
        #     res.append({"params": layer.parameters(), "lr": clr})
        #     clr *= d

        # Linear:
        d = float(upper - lower) / float(group)
        gl = int(len(self.main_model._blocks) / group)
        clr = lower

        tmp_param = []
        tmp_param += list(self.main_model._conv_stem.parameters())
        tmp_param += list(self.main_model._bn0.parameters())

        for inx, layer in enumerate(self.main_model._blocks):
            tmp_param += list(layer.parameters())
            if (inx + 1) % gl == 0:
                res.append({"params": tmp_param, "lr": clr})
                tmp_param = []
                clr += d

        tmp_param += list(self.main_model._conv_head.parameters())
        tmp_param += list(self.main_model._bn1.parameters())
        tmp_param += list(self.main_model._fc.parameters())
        tmp_param += list(self.heads_pool.parameters())
        tmp_param += list(self.heads.parameters())
        res.append({"params": tmp_param, "lr": upper})

        # for layer in [model.main_model._conv_head, model.main_model._bn1]:
        #     res.append({"params": layer.parameters(), "lr": upper})

        # res.append({"params": model.main_model._fc.parameters(), "lr": upper})

        ######################################################################3

        # for layer in [model.heads_pool, model.heads]:
        #     res.append({"params": layer.parameters(), "lr": upper})

        return res
