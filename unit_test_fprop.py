import argparse
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from praxis import pax_fiddle
from praxis import base_layer, layers
from praxis.layers.injection import fp8_nvidia_gpu as fp8_ops

HAVE_FP8_OP = True

maybe_shard = base_layer.maybe_shard
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

class TestModel(base_layer.BaseLayer):
    dtype: jnp.dtype = jnp.bfloat16 
    hidden_size: int = 0
    einsum_tpl: LayerTpl = template_field(layers.base_ops.EinsumOp)
    gelu_tpl: LayerTpl = template_field(layers.GELU)

    def setup(self):
        self.create_child('fc1', self.einsum_tpl.clone())
        self.create_child('gelu', self.gelu_tpl.clone())
        self.create_child('fc2', self.einsum_tpl.clone())

    def __call__(self, inputs, weights1, weights2):
        weights1 = jnp.asarray(weights1, self.dtype)
        weights2 = jnp.asarray(weights2, self.dtype)
        output = self.fc1('...y,yz->...z', inputs, weights1)
        output = self.gelu(output)
        output = self.fc2('...y,yz->...z', output, weights2)
        return output

def train_step(model, params, inputs, weights1, weights2, grad):
    out, f_vjp = jax.vjp(model.apply, params, inputs, weights1, weights2)
    return out

def main():
    parser = argparse.ArgumentParser(description='MLP Unit Test with pjit')
    parser.add_argument("--dp", dest="dp", type=int, default=1)
    parser.add_argument("--zp", dest="zp", type=int, default=1)
    parser.add_argument("--tp", dest="tp", type=int, default=4)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2) #GPT-175B FFN shape
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=2048) #GPT-175B FFN shape
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=12288) #GPT-175B FFN shape
    parser.add_argument("--use_fp8", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=True)
    args = parser.parse_args()

    assert(args.dp * args.zp * args.tp == len(list(jax.devices())))
    args.batch_size = args.batch_size * args.dp * args.zp
    if args.use_fp8:
        assert HAVE_FP8_OP, "FP8 OPs are not defined."

    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    inputs = jax.random.uniform(key2, (args.batch_size, args.seq_len, args.hidden_size), dtype=dtype)
    weights1 = jax.random.uniform(key2, (args.hidden_size, 4*args.hidden_size), dtype=jnp.float32)
    weights2 = jax.random.uniform(key2, (4*args.hidden_size, args.hidden_size), dtype=jnp.float32)
    grad = jax.random.uniform(key2, (args.batch_size, args.seq_len, args.hidden_size), dtype=dtype)

    model = TestModel(dtype=dtype, hidden_size=args.hidden_size)
    if args.use_fp8:
        model.einsum_tpl = pax_fiddle.Config(fp8_ops.Fp8EinsumOp)
    params = model.init(key1, inputs, weights1, weights2)
    
    mesh_shape = {'dp': args.dp, 'zp': args.zp, 'tp': args.tp}
    mesh = Mesh(np.array(jax.devices()).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))

    pjitted_train_step = pjit(partial(train_step, model),
                              out_shardings=PartitionSpec(('dp', 'zp'), None, 'tp'))

    if args.profile:
        import ctypes
        libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
        with mesh:
            for i in range(100):
                if i == 9:
                    libcudart.cudaProfilerStart()
                inputs = maybe_shard(inputs, [('dp', 'zp'), 'tp', None], tuple(mesh_shape.keys()))
                weights1 = maybe_shard(weights1, ['zp', 'tp'], tuple(mesh_shape.keys()))
                weights2 = maybe_shard(weights2, ['tp', 'zp'], tuple(mesh_shape.keys()))
                grad = maybe_shard(grad, [('dp', 'zp'), 'tp', None], tuple(mesh_shape.keys()))
                out = pjitted_train_step(params, inputs, weights1, weights2, grad)
                if i == 14:
                    libcudart.cudaProfilerStop()
    else:
        with mesh:
            for i in range(100):
                inputs = maybe_shard(inputs, [('dp', 'zp'), 'tp', None], tuple(mesh_shape.keys()))
                weights1 = maybe_shard(weights1, ['zp', 'tp'], tuple(mesh_shape.keys()))
                weights2 = maybe_shard(weights2, ['tp', 'zp'], tuple(mesh_shape.keys()))
                grad = maybe_shard(grad, [('dp', 'zp'), 'tp', None], tuple(mesh_shape.keys()))
                out = pjitted_train_step(params, inputs, weights1, weights2, grad)

    return out

if __name__ == "__main__":
    main()
