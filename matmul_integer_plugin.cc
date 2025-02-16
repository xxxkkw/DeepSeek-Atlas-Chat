#include "register/register.h"
namespace domi {
    Status ParseParamsMatmulInteger(const ge::Operator& op_src, ge::Operator& op_dest) {
        return SUCCESS;
    }
    REGISTER_CUSTOM_OP("BatchMatMulV2")
        .FrameworkType(ONNX)
        .OriginOpType({ge::AscendString("ai.onnx::10::MatMulInteger"),
                       ge::AscendString("ai.onnx::11::MatMulInteger"),
                       ge::AscendString("ai.onnx::12::MatMulInteger"),
                       ge::AscendString("ai.onnx::13::MatMulInteger"),
                       ge::AscendString("ai.onnx::14::MatMulInteger"),
                       ge::AscendString("ai.onnx::15::MatMulInteger")})
        .ParseParamsByOperatorFn(ParseParamsMatmulInteger)
        .ImplyType(ImplyType::TVM);
}
