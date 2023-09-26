import onnx_opslist
def RnnRWKV(ops: onnx_opslist.RWKVOnnxOps, *args):
    class myRWKV(ops.module):

        @ops.initfunc
        def __init__(self, version, w, seq_length=16, converted=False):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")

            self.version = version
            self.ops = ops
            self.seq_length = seq_length
            self.postprocess0 = ops.initTensor((w["ln_out.weight"]))
            self.postprocess1 = ops.initTensor((w["ln_out.bias"]))
            self.postprocess2 = ops.initTensor((w["head.weight"]))
            self.emb = ops.initTensor(w["emb.weight"])
            if not converted:
                self.emb1 = ops.initTensor(w["blocks.0.ln0.weight"])
                self.emb2 = ops.initTensor(w["blocks.0.ln0.bias"])
            self.lx_w = (ops.stack(
                [w[f"blocks.{x}.att.ln_x.weight"] for x in range(ops.n_layers)]))
            self.lx_b = (ops.stack(
                [w[f"blocks.{x}.att.ln_x.bias"] for x in range(ops.n_layers)]))
            self.ln1w = (ops.stack(
                [w[f"blocks.{x}.ln1.weight"] for x in range(ops.n_layers)]))
            self.ln1b = (ops.stack(
                [w[f"blocks.{x}.ln1.bias"] for x in range(ops.n_layers)]))
            self.ln2w = (ops.stack(
                [w[f"blocks.{x}.ln2.weight"] for x in range(ops.n_layers)]))
            self.ln2b = (ops.stack(
                [w[f"blocks.{x}.ln2.bias"] for x in range(ops.n_layers)]))
            if not converted:
                self.time_decay = (ops.stack([
                    w[f"blocks.{x}.att.time_decay"].double().exp().neg() for x in range(ops.n_layers)], True))
            else:
                self.time_decay = (ops.stack([
                    w[f"blocks.{x}.att.time_decay"].double() for x in range(ops.n_layers)], True))
            self.time_first = (ops.stack([
                w[f"blocks.{x}.att.time_first"] for x in range(ops.n_layers)], True))
            self.kktk = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_k"] for x in range(ops.n_layers)]))
            self.vvtv = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_v"] for x in range(ops.n_layers)]))
            self.rrtr = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_r"] for x in range(ops.n_layers)]))
            if version == 5.1:
                self.ggtg = (ops.stack(
                    [w[f"blocks.{x}.att.time_mix_g"] for x in range(ops.n_layers)]))
            self.key = (ops.stack(
                [w[f"blocks.{x}.att.key.weight"] for x in range(ops.n_layers)], exname="_key"))
            self.value = (ops.stack(
                [w[f"blocks.{x}.att.value.weight"] for x in range(ops.n_layers)], exname="_value"))
            self.receptance = (ops.stack([
                w[f"blocks.{x}.att.receptance.weight"] for x in range(ops.n_layers)], exname="_receptance"))
            if version == 5.1:
                self.gate = (ops.stack([
                    w[f"blocks.{x}.att.gate.weight"] for x in range(ops.n_layers)], exname="_gate"))
            self.outputvv = (ops.stack([
                w[f"blocks.{x}.att.output.weight"] for x in range(ops.n_layers)], exname="_outputvv"))
            self.time_mix_k_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_k"] for x in range(ops.n_layers)]))
            self.time_mix_r_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_r"] for x in range(ops.n_layers)]))
            self.key_ffn = (ops.stack(
                [w[f"blocks.{x}.ffn.key.weight"] for x in range(ops.n_layers)], exname="_key_ffn"))
            self.receptance_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.receptance.weight"] for x in range(ops.n_layers)], exname="_receptance_ffn"))
            self.value_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.value.weight"] for x in range(ops.n_layers)], exname="_value_ffn"))
            self.w = w;

        def wkv(self, k, v, xx, statee, stateb, statec):
            ww = ops.add(k, self.time_first[xx])
            p = ops.maximum(statee, ww)

            e1 = ops.exp(ops.subtract(statee, p))
            e2 = ops.exp(ops.subtract(ww, p))
            a = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v))
            b = ops.add(ops.multiply(e1, statec), e2)
            ww = ops.add(statee, self.time_decay[xx])

            p = ops.maximum(ww, k)

            e1 = ops.exp(ops.subtract(ww, p))
            e2 = ops.exp(ops.subtract(k, p))
            outb = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v))
            outc = ops.add(ops.multiply(e1, statec), e2)
            eee = p
            wkv = ops.divide(a, b)

            return ops.convertToFloat16(wkv), outb, outc, eee

        @ops.layerdef
        def doLayer(self, x, statea, stateb, statec, stated, statee, xx):

            xy = ops.layernorm(x, self.ln1w[xx], self.ln1b[xx])

            # Time Mixing
            if self.version == 4:
                k = ops.matvec(
                    self.key[xx], ops.lerp(statea, xy, self.kktk[xx]), True)

                v = ops.matvec(self.value[xx], ops.lerp(
                    statea, xy, self.vvtv[xx]), True)
                rr = ops.matvec(
                    self.receptance[xx], ops.lerp(statea, xy, self.rrtr[xx]))
                r = ops.logistical((rr))

                wkv, outb, outc, eee = self.wkv(k, v, xx, statee, stateb, statec)

                mvv = ops.add(x, ops.matvec(
                    self.outputvv[xx], ops.multiply(r, wkv)))
                state_ffn = stated
            else:
                assert self.version == 5.1
                # auto H = t_decay.size(0);
                # auto S = x.size(x.shape().size() - 1) / H;
                H = self.w[f"blocks.{xx}.att.time_decay"].size(0)
                assert self.w[f"blocks.{xx}.att.ln_x.weight"].size(0) % H == 0
                S = self.w[f"blocks.{xx}.att.ln_x.weight"].size(0) // H

                k = ops.matvec(
                    ops.lerp(statea, xy, self.kktk[xx]), self.key[xx], True)
                k = ops.reshape(k, (H, S, 1))

                v = ops.matvec(ops.lerp(
                    statea, xy, self.vvtv[xx]), self.value[xx], True)
                v = ops.reshape(v, (H, 1, S))
                r = ops.matvec(
                    ops.lerp(statea, xy, self.rrtr[xx]), self.receptance[xx])
                r = ops.reshape(r, (H, 1, S))
                g = ops.matvec(
                    ops.lerp(statea, xy, self.ggtg[xx]), self.gate[xx])
                g = ops.multiply(ops.logistical(g), g)

                a = ops.matvec(k, v)
                out = ops.matvec(r, ops.add(ops.multiply(self.time_first[xx], a), stateb) )
                decayed_s = ops.add(ops.multiply(self.time_decay[xx], stateb), a)

                out = ops.pytorch_flatten(out)
                out = ops.groupnorm(ops.unsqueeze(out, 0), self.lx_w[xx], self.lx_b[xx], H, 1e-5)
                out = ops.pytorch_flatten(out)
                out = ops.multiply(g, out)

                mvv = ops.add(x, ops.matvec(out, self.outputvv[xx]))
                state_ffn = statec

            # Channel Mixing
            ddd = ops.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            km = ops.relu(ops.matvec(ops.lerp(
                state_ffn, ddd, self.time_mix_k_ffn[xx]), self.key_ffn[xx]))

            rt = ops.logistical((ops.matvec(ops.lerp(
                state_ffn, ddd, self.time_mix_r_ffn[xx]), self.receptance_ffn[xx])))

            x = ops.add(mvv, ops.multiply(
                ops.matvec(ops.multiply(km, km), self.value_ffn[xx]), rt))

            if self.version == 5.1:
                return x, ops.convertToFloat32(xy), decayed_s, ops.convertToFloat32(ddd)
            else:
                return x, ops.convertToFloat32(xy), outb, outc, ops.convertToFloat32(ddd), eee # why not eee, ops.convertToFloat32(ddd)

        @ops.layerdef
        def doSeqLayer(self, x, statea, stateb, statec, stated, statee, i):

            # Time Mixing
            xx = ops.layernorm(x, self.ln1w[i], self.ln1b[i])
            statea = ops.concate(ops.unsqueeze(statea, 0), ops.slice(xx, axes=0, starts=0, ends=-1))
            k = ops.matvec(
                ops.lerp(statea, xx, self.kktk[i]),
                self.key[i],
                True
            )
            v = ops.matvec(
                ops.lerp(statea, xx, self.vvtv[i]),
                self.value[i],
                True
            )
            rr = ops.matvec(
                ops.lerp(statea, xx, self.rrtr[i]),
                self.receptance[i],
                True
            )
            r = ops.logistical((rr))
            state_list = []
            for t in range(self.seq_length):
                kk = ops.squeeze(ops.slice(k, axes=0, starts=t, ends=t+1))
                vv = ops.squeeze(ops.slice(v, axes=0, starts=t, ends=t+1))
                temp, stateb, statec, statee = self.wkv(kk, vv, i, statee, stateb, statec)
                state_list.append(ops.unsqueeze(temp))
            statea = ops.seq_concate(state_list, self.seq_length)
            mvv = ops.add(x, ops.matvec(ops.multiply(r, statea), self.outputvv[i]))
            ret1 = ops.slice(xx, axes=0, starts=-1, ends=65535)#-2????

            # Channel Mixing
            ddd = ops.layernorm(mvv, self.ln2w[i], self.ln2b[i])
            stated = ops.concate(ops.unsqueeze(stated, 0), ops.slice(ddd, axes=0, starts=0, ends=-1))
            km = ops.relu(ops.matvec(ops.lerp(stated, ddd, self.time_mix_k_ffn[i]), self.key_ffn[i]))
            rt = ops.logistical((ops.matvec(ops.lerp(stated, ddd, self.time_mix_r_ffn[i]), self.receptance_ffn[i])))
            x = ops.add(mvv, ops.multiply(
                ops.matvec(ops.multiply(km, km), self.value_ffn[i]), rt))
            ret2 = ops.slice(ddd, axes=0, starts=-1, ends=65535)#-2????
            # why not eee, ops.convertToFloat32(ret2)
            return x, ops.convertToFloat32(ret1), stateb, statec, ops.convertToFloat32(ret2), statee


        @ ops.mainfunc
        def forwardSeq(self, x, state = None):
            if (state is None):
                state = ops.emptyState
            if converted:
                x = ops.getIndex(self.emb, x)
            else:
                x = ops.layernorm(
                    ops.getIndex(self.emb, x),
                    self.emb1, self.emb2)

            statea = state[0::5]
            stateb = state[1::5]
            statec = state[2::5]
            stated = state[3::5]
            statee = state[4::5]

            ot = []
            for i in range(ops.n_layers):
                x, aaa, bbb, ccc, ddd, eee = self.doSeqLayer(
                    x,
                    ops.convertToFloat16(statea[i]),
                    (stateb[i]),
                    (statec[i]),
                    ops.convertToFloat16(stated[i]),
                    (statee[i]),
                    i
                )
                ot = ot + [aaa, bbb, ccc, ddd, eee]
            x = ops.matvec(ops.layernorm(x, self.postprocess0, self.postprocess1),
                           self.postprocess2)
            return ops.convertToFloat32(x), ot


        @ ops.mainfunc
        def forward(self, x, state = None):
            if (state is None):
                state = ops.emptyState
            if converted:
                x = ops.getIndex(self.emb, x)
            else:
                x = ops.layernorm(
                    ops.getIndex(self.emb, x),
                    self.emb1, self.emb2)
            if self.version == 4:
                statea = state[0::5]
                stateb = state[1::5]
                statec = state[2::5]
                stated = state[3::5]
                statee = state[4::5]

                ot = []
                for i in range(ops.n_layers):
                    x, aaa, bbb, ccc, ddd, eee = self.doLayer(
                        x,
                        ops.convertToFloat16(statea[i]),
                        (stateb[i]),
                        (statec[i]),
                        ops.convertToFloat16(stated[i]),
                        (statee[i]),
                        i
                    )

                    ot = ot + [aaa, bbb, ccc, ddd, eee]

            else:
                assert self.version == 5.1
                statea = state[0::3]
                stateb = state[1::3]
                statec = state[2::3]
                ot = []
                for i in range(ops.n_layers):
                    x, aaa, bbb, ccc = self.doLayer(
                        x,
                        ops.convertToFloat16(statea[i]),
                        (stateb[i]),
                        (statec[i]),
                        None,
                        None,
                        i
                    )

                    ot = ot + [aaa, bbb, ccc]


            x = ops.matvec(ops.layernorm(x, self.postprocess0, self.postprocess1), self.postprocess2)
            return ops.convertToFloat32(x), ot


    ops.postProcessModule(myRWKV(*args))



import torch

def convert_model(path, dtype, version):
    w = torch.load(path, map_location="cpu")
    dims = len(w["blocks.0.att.key.weight"])
    layers = len(list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))


    ops = onnx_opslist.RWKVOnnxOps(
        layers,
        dims,
        dtype=dtype,
        opsVersion=opset_version,
        externalData=use_external_data,
        splitExternalData=splitExternalData,
        fp32inout=fp32inout,
        seq_mode=seq_mode,
        seq_length=seq_length
    )

    RnnRWKV(ops, version, w, seq_length, converted)


import numpy as np
def convert():
    path = input_path
    dtype = np.float16 if use_fp16 else np.float32
    convert_model(path, dtype, 5.1)

# Define the variables
input_path = r"RWKV-5-MIDI-560M-v1-20230902-ctx4096-fp32-converted.pth"
if "convert" in input_path:
    converted = True
else:
    converted = False
assert converted
use_fp16 = False
use_external_data = True
splitExternalData = False
fp32inout = True
# opset version, number either 15/17
opset_version = 17

# set seq mode and length
seq_mode = False
seq_length = 16
convert()
