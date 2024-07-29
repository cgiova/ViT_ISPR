import torch
from x_transformers import Encoder, ViTransformerWrapper, ViTransformerWrapperMod

# set hyperparams
parameters = {
    'image_size': 256,
    'patch_size': 32,
    'num_classes': 1000,
    'attn_layers': Encoder(
        dim=512,
        depth=6,
        heads=8
    )
}

def create_img():
    return torch.randn(10, 3, 256, 256)


def create_transformer_wrapper(params):
    return ViTransformerWrapper(**params)


def create_transformer_wrapper_mod(params):
    return ViTransformerWrapperMod(**params)


# define testing script
def test_transformer_wrapper():
    model = create_transformer_wrapper(parameters)
    img = create_img()
    model(img)


def test_transformer_wrapper_mod():
    model = create_transformer_wrapper_mod(parameters)
    img = create_img()
    model(img)


# Run the test
if __name__ == "__main__":
    test_transformer_wrapper()
    print("="*50)
    test_transformer_wrapper_mod()
