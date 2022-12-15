import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2


num_layers = 5
num_heads = 6
embedding_size = 1024
num_words = 64
num_patches = 64
pic_size = 256

# model hyper-parameters
learning_rate = .0001
batch_size = 2


class PositionwiseFeedForward(nn.Module):
    "Simple linear layers with dropout and relu"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x.float()))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x.float()), dim=-1)


class LayerNorm(nn.Module):
    "Construct a layernorm module "

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # return x  # todo: we commented this out.  May not be a good idea


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


"""## Encoder

The encoder is composed of a stack of $N=6$ identical layers. 
"""


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward "

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


"""## Decoder

The decoder is also composed of a stack of $N=6$ identical layers.  

"""


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return (x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x,
                             lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


"""##Implement Attention


https://arxiv.org/pdf/1706.03762.pdf         

"""

my_softmax = nn.Softmax(dim=2)


def attention(query, key, value, mask):
    # Compute 'Scaled Dot Product Attention'
    scores = torch.matmul(query, torch.transpose(key, 1, 2))
    # scores = QK^T/scale
    scores /= np.sqrt(key.size()[2])
    # Apply the mask
    if mask is not None:
        if mask.device.type != 'cuda':
            mask = mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)

    output = my_softmax(scores)

    output = torch.matmul(output, value)

    return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # Implement Multi-head attention mechanism
        self.values_linears = clones(nn.Linear(d_model, d_model), h)
        self.queries_linears = clones(nn.Linear(d_model, d_model), h)
        self.keys_linears = clones(nn.Linear(d_model, d_model), h)
        # Make an attention head (linear layers for q, k, and v)
        # head = values_linear, queries_linear, keys_linear
        # Make h copies of the attention head (Hint: See the `clone()` helper function)
        # self.heads = clones(head, h)
        self.h = h
        self.final_linear = nn.Linear(h * d_model, d_model)

    def forward(self, query, key, value, mask):
        # For each attention head
        # Pass the query, key, value through their respective layers
        # Compute scaled dot-product attention on the output
        outputs = []
        # query = query.type('torch.DoubleTensor').cuda()
        # key = key.type('torch.DoubleTensor').cuda()
        # value = value.type('torch.DoubleTensor').cuda()

        for i in range(self.h):
            outputs += [attention(self.queries_linears[i](query.float()),
                                  self.keys_linears[i](key.float()),
                                  self.values_linears[i](value.float()), mask)]
        output = torch.cat(outputs, 2)
        output = self.final_linear(output)
        return output


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = 1 / (10000 ** (torch.arange(0., d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

    def subtract_embedding(self, x):
        x = x - torch.transpose(Variable(self.pe[:, :x.size(2)],
                                         requires_grad=False), 2, 1)
        return x



class TransformerModel(nn.Module):
    """
    Full transformer model
    """

    def __init__(self, N=4, d_model=256, d_ff=256,
                 h=2, dropout=0.1):
        super(TransformerModel, self).__init__()

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout, max_len=num_words)
        c = copy.deepcopy

        self.encoder = Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                            c(ff), dropout), N)

        self.end_linear1 = nn.Linear(d_model, d_model)
        self.end_linear2 = nn.Linear(d_model, d_ff)
        self.end_linear3 = nn.Linear(d_ff, d_model)


        self.t_conv1 = nn.ConvTranspose2d(1, 64, 3, stride=1, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 16, 3, stride=1, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)

        # self.generator = Generator(d_model, num_words)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        # x = self.encode(src, src_mask)
        # x = self.end_linear2(self.end_linear1(x))
        # x = self.reformat_picture_model(x, 16, 16)
        # x = x.unsqueeze(1)
        # x = F.relu(self.t_conv1(x))
        # x = F.relu(self.t_conv2(x))
        # x = F.relu(self.t_conv3(x))
        # x = x.squeeze()
        # x = self.end_linear3(x)
        # return x
        return self.end_linear1(self.decode(self.encode(src, src_mask), src_mask,
                                                                              tgt,
                                                                              tgt_mask))  # todo: need triangle mask
        # return self.position.subtract_embedding(self.decode(self.encode(src, src_mask), src_mask,
        #                    tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.position(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.position(tgt), memory, src_mask,
                            tgt_mask)


class VisualTransformerModel(nn.Module):
    """ Full Model """

    def __init__(self, N=4, d_model=256, d_ff=256, h=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.transformer_model = TransformerModel(N, d_model, d_ff,
                                                  h, dropout)
        self.start_token = torch.nn.Linear(d_model, d_model, bias=False).cuda()
        self.first_word_mask = torch.zeros((batch_size, d_ff, d_model))
        self.first_word_mask[:, 0, :] = torch.ones((batch_size, d_model))
        self.first_word_mask.cuda()
        self.num_layers = 5
        self.num_heads = 6
        self.embedding_size = 1024
        self.num_words = 64
        self.num_patches = 8
        self.pic_size = 256
        self.d_ff = d_ff
        self.d_model = d_model
        # self.generator = Generator(d_model, d_model)

    def pic2words(self, x):
        words = F.unfold(x, kernel_size=int(np.sqrt(self.embedding_size)), stride=int(np.sqrt(self.embedding_size)))
        words = torch.transpose(words, 2, 1)
        # plt.imshow(words[0,:,:].cpu().detach().numpy())
        # plt.show()
        return words

    def words2pic(self, x):
        x = torch.transpose(x, 2, 1)
        pic = F.fold(x, output_size=256, kernel_size=int(np.sqrt(self.embedding_size)), stride=int(np.sqrt(self.embedding_size)))
        return pic

    def forward(self, in_batch):
        x = in_batch[:,0,:,:]
        x = x.unsqueeze(1)
        batch = self.pic2words(x)
        batch_dim0 = batch.shape[0]
        output_tensor = torch.cat((torch.ones((batch_dim0, 1, batch.shape[2])).cuda(), batch.cuda()), dim=1)

        batch = Batch(batch, output_tensor, pad=-1)
        src = batch.src
        tgt = batch.trg
        src_mask = batch.src_mask
        tgt_mask = batch.trg_mask

        "Take in and process masked src and target sequences."
        if batch_dim0 < batch_size:
            self.first_word_mask_small = torch.zeros((batch_dim0, self.d_ff, self.d_model))
            self.first_word_mask_small[:, 0, :] = torch.ones((batch_dim0, self.d_model))
            self.first_word_mask_small.cuda()
            start_token = (self.start_token(
                torch.ones((batch_dim0, 1, self.d_model)).cuda()) * self.first_word_mask_small.cuda())
        else:
            start_token = (self.start_token(torch.ones((batch_dim0, 1, self.d_model)).cuda()) * self.first_word_mask.cuda())
        tgt += start_token
        out = self.transformer_model.forward(src, tgt, src_mask, tgt_mask)
        out = self.words2pic(out)
        out = torch.squeeze(out)
        out = torch.stack((out,out,out), dim=1)
        return out


"""# Training

## Batches and Masking
"""

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        # self.src_mask = np.expand_dims((src != pad), -2) #.unsqueeze(-2)
        # self.src_mask = (src != pad)#.unsqueeze(-2)
        self.src_mask = (torch.ones((src.shape[0],src.shape[1],src.shape[1])) > 0)
        self.src_mask = self.src_mask
        if trg is not None:
            # self.trg = trg[:, :,:-1]
            # self.trg_y = trg[:, :,1:]
            # trg_mask should be batch_size, num_words, num_words
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            # self.trg_mask = ((torch.ones((trg.shape[0],trg.shape[1] - 1,trg.shape[1] - 1)) > 0))
            # todo: add diagonal mask back
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt_mask = np.expand_dims((tgt != pad),-2) #.unsqueeze(-2)
        tgt_mask = (tgt != pad).unsqueeze(-2)

        # tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.shape[-1]).cuda())
        tgt_mask = Variable(subsequent_mask(tgt.size(1)).type_as(tgt_mask.data))
        return tgt_mask


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


"""## Label Smoothing

During training, we employed label smoothing of value $\epsilon_{ls}=0.1$ [(cite)](https://arxiv.org/abs/1512.00567).  This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.  
"""


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # x = F.sigmoid(self.t_conv2(x))

        return x


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        # TODO: Double check these two lines. I don't think they are working right.
        true_dist = true_dist.flatten().unsqueeze(-1)
        true_dist.scatter_(0, target.data.unsqueeze(-1).type(torch.int64), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(torch.Tensor(target.data == self.padding_idx))
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


"""## Data Loading

"""


"""## Training Code"""


class LossFunction:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        # self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x = self.generator(x.float())
        # loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                       y.contiguous().view(-1)) / norm
        loss = self.criterion(x.float(), y.float())
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data


def reformat_picture(img_tensor, num_sections, patch_size):
    patches = []
    img = img_tensor.cpu().detach().numpy()

    y, x = img.shape
    step_y = y // num_sections
    step_x = x // num_sections
    reconstructed_img = np.zeros((num_sections * patch_size, num_sections * patch_size))

    for i in range(num_sections ** 2):
        patches += [img[i, :].reshape((patch_size, patch_size))]
        reconstructed_img[int(i / num_sections) * patch_size:int(i / num_sections) * patch_size + patch_size,
        (i % num_sections) * patch_size:((i) % num_sections) * patch_size + patch_size] = patches[i]

    # cv2.imshow('test', reconstructed_image[0])
    # cv2.waitKey()
    return reconstructed_img


# model = VisualTransformerModel(num_layers, embedding_size, num_words, num_heads).cuda()
# model_opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# input = np.arange(10*3*256*256).reshape((10,3,256,256))
# plt.imshow(input[0,0,:,:])
# plt.colorbar()
# plt.show()
# out = model.forward(torch.Tensor(input).cuda())



