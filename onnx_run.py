import numpy as np
import os

def initONNXFile(path, useAllAvailableProviders=False):
    import onnxruntime as rt

    # session execution provider options
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = rt.InferenceSession(
        path, sess_options)#, providers="CPUExecutionProvider")

    ins = {

    }
    path = os.path.basename(path)
    embed = int(path.split("_")[2].split(".")[0])
    layers = int(path.split("_")[1])
    typenum = sess.get_inputs()[1].type
    # typenum = sess.get_inputs()[1].type
    print(typenum)
    import numpy as np

    if typenum == "tensor(float)":
        typenum = np.float32
    elif typenum == "tensor(float16)":
        typenum = np.float16

    class InterOp():

        RnnOnly = True

        def forward(self, xi, statei):
            # print(statei[0][23])
            # create inputs
            inputs = ins
            # get input names
            input_names = sess.get_inputs()
            input_names = [x.name for x in input_names]
            # get output names
            output_names = sess.get_outputs()
            output_names = [x.name for x in output_names]
            # print(output_names)

            # create input dict
            inputs[input_names[0]] = np.array(xi, dtype=np.int32)
            for i in range(len(input_names)-1):
                inputs[input_names[i+1]] = statei[i]

            outputs = sess.run(output_names, inputs)
            # print(outputs[1][23])

            return outputs[0], outputs[1:]

    model = InterOp()

    version = 5.1
    # emptyState = []
    # emptyState = np.array((([[0.01]*embed, [0.01]*embed, [0.01]*embed, [
    #         0.01]*embed]+([[-1e30]*embed] if not isUnsafeWKV else [])))*layers, typenum)
    
    if version == 5.1:
        # TODO:
        emptyState = []
        for i in range(layers):
            emptyState.append(np.zeros((embed,), dtype=typenum))
            # NOTE: hardcode
            emptyState.append(np.zeros((14,64,64), dtype=typenum))
            emptyState.append(np.zeros((embed,), dtype=typenum))


    return model, emptyState

def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    import numpy as np
    from scipy.special import softmax

    try:
        ozut = ozut.numpy()
    except:
        try:
            ozut = ozut.cpu().numpy()
        except:
            ozut = np.array(ozut)
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = pow(probs, 1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    # mout = np.argmax(probs).astype(np.int32)
    mout = np.random.choice(a=len(probs), p=probs)
    return mout

########
# model, state = initONNXFile("RWKV-5-ABC-82M-v1-20230901-ctx1024-sim.onnx")
model, state = initONNXFile("./RWKV_53_896_32_17.onnx")
# model_seq, state_seq = initONNXFile("./RWKV_24_2048_32_17_seq.onnx")


# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("./gpt-neox-20b", pad_token='<|padding|>')
# tokenizer.pad_token = '<|padding|>'

# context = "Hi, who are you?"
# print(context, "\n")
#
# # seq
# prompt = tokenizer.encode(context, padding="max_length", max_length=16, pad_to_max_length=True)
#
# logits, state = model_seq.forward(prompt, state_seq)
# state = np.vstack(state)
# logits = logits[-1,:]


# prompt = tokenizer.encode(context)
prompt = [2]
for token in prompt:
    logits, state = model.forward([token],state)
    import pdb; pdb.set_trace()


# normal
for i in range(1000):
    prompt = prompt+[npsample(logits, top_p_usual=0)]
    print(chr(prompt[-1]),end="", flush=True)
    logits, state = model.forward([prompt[-1]],state)
print("\n")

