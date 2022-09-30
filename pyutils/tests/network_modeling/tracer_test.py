import torch
from torch import fx
from torch.fx import Tracer


class Custom(Tracer):
    def __init__(self):
        super().__init__()


def transform(m: torch.nn.Module, tracer_class: type = Custom) -> torch.nn.Module:
    graph: fx.Graph = tracer_class().trace(m)
    # for node in graph.nodes:
    #     # print(node.op, node.target)
    graph.lint()
    return fx.GraphModule(m, graph)

# def main(nw):
#     # nw = "inception"        # has fusion group
#     bm = get_unified_benchmark_name(nw)
#     in_t = ModelsAndInputs.get_example_cuda(model_name=bm, batch_size=1)
#     model = ModelsAndInputs._get_model(bm).cuda().eval()
#
#     # torch.cudnn_convolution_relu()
#     # torch.conv2d()
#     with torch.jit.fuser('fuser1'):
#         # torch._C._jit_override_can_fuse_on_gpu(False)
#         with torch.no_grad():
#             with torch.jit.optimized_execution(True):
#                 model = torch.jit.trace(model, in_t)
#                 # model = torch.jit.script(model)
#                 model = torch.jit.freeze(model)
#                 model = torch.jit.optimize_for_inference(model)
#                 # print(model.inlined_graph)
#                 # print(model.graph_for(in_t))
#                 # model = torch.jit.script(model)
#                 # t = transform(model, Custom)
#                 model(in_t)
#                 model(in_t)
#                 model(in_t)
#                 k = torch.jit.last_executed_optimized_graph()
#                 ops = torch.jit.export_opnames(model)
# print(k)
# print(ops)
# # print(k)
# # m.save('graph.pt')
#
# print(nw, ops)
# _extract_tensors(m)
# m = torch.propagate_and_assign_input_shapes()
# print(dir(m))

# # print(dir(torch._C.Node))
# # print(m.param_node())
# for node in k.nodes():
#     if node.schema() == '(no schema)':
#         continue
#     # # #     print()
#     #     print(dir(node))
#     print(dir(node.outputsAt(0)))
#     print(node.outputsAt(0).type())
#     print(node.outputsAt(0).toIValue())
#     print(node.outputsAt(0).debugName())
#     print(node.outputsAt(0).__repr__())
#     print(node.kind())
#     print(node.schema())
#     print(dir(node))
# #     # print(node.attributeNames())
# #     # print(node.inputsSize())
#     # print(node.scalar_args)
#     for j in node.inputs():
#         print(j)
# #
#         #['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
#         # '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',
#         # '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
#         # '__sizeof__', '__str__', '__subclasshook__', 'copyMetadata', 'debugName', 'inferTypeFrom',
#         # 'isCompleteTensor', 'node', 'offset', 'replaceAllUsesAfterNodeWith', 'replaceAllUsesWith',
#         # 'requiresGrad', 'requires_grad', 'setDebugName', 'setType', 'setTypeAs', 'toIValue', 'type',
#         # 'unique', 'uses']
#         # print(dir(j))
#         # print(j)
#
#         print(j.debugName())
#         print(node.__getattribute__(j.debugName()))
#         l = j.toIValue()
#         print(l)
#     print()
# print([i.inputs() for i in m.nodes()])
# # print([i.input() for i in m.nodes()])
# print([i.inputsSize() for i in m.nodes()])
# print(type(m))
# print(dir(torch._C.Graph))

#
# for _nw in ["mnasnet"]:
#     if _nw != "googlenet":
#         main(_nw)
