#include "register/register.h"
namespace domi {
    Status ParseParamsDequantizeLinear(const ge::Operator& op_src, ge::Operator& op_dest) {
        return SUCCESS;
    }
    REGISTER_CUSTOM_OP("AscendAntiQuant")
        .FrameworkType(ONNX)
        .OriginOpType({ge::AscendString("ai.onnx::10::DequantizeLinear"),
                       ge::AscendString("ai.onnx::11::DequantizeLinear"),
                       ge::AscendString("ai.onnx::12::DequantizeLinear"),
                       ge::AscendString("ai.onnx::13::DequantizeLinear"),
                       ge::AscendString("ai.onnx::14::DequantizeLinear"),
                       ge::AscendString("ai.onnx::15::DequantizeLinear")})
        .ParseParamsByOperatorFn(ParseParamsDequantizeLinear)
        .ImplyType(ImplyType::TVM);
}
