# Image Captioning: CNN Encoder + Attention-LSTM / Transformer Decoder

An image captioning system that combines a pre-trained CNN for visual feature
extraction with a sequence model (LSTM with attention, or a Transformer
decoder) to generate natural-language captions.

## Architecture

Image (224x224x3)
   |
Pretrained CNN (ResNet50 or VGG16, ImageNet weights, frozen by default)
   |
Spatial feature grid: 14x14x2048 -> reshaped to 196 x 2048
   |
   +-----------------------------------+
   |                                   |
LSTM decoder w/ Bahdanau attention     Transformer decoder
(models/model.py) <-- default          (models/transformer_model.py) <-- bonus
   |                                   |
   +-----------------------------------+
   |
Generated caption, one word at a time
(beam search or greedy at inference)

**Why this design:**
- **CNN encoder**: ResNet50 / VGG16 pretrained on ImageNet already encode
  rich visual concepts (objects, textures, scenes). We keep the last
  convolutional feature *map* (not the final pooled vector) so the decoder
  can look at different spatial regions of the image as it generates each
  word — this is what makes the attention mechanism meaningful.
- **LSTM + attention decoder** (default, `models/model.py`): implements
  "Show, Attend and Tell" — at every decoding step, additive attention
  computes a weighted combination of the 196 spatial feature vectors,
  conditioned on the decoder's current hidden state.
- **Transformer decoder** (bonus, `models/transformer_model.py`): treats the
  CNN feature grid as cross-attention memory, using `nn.TransformerDecoder`
  with causal self-attention over the caption so far.

## Setup
