#include "register/register.h"
namespace domi {
    Status ParseParamsDynamicQuantizeLinear(const ge::Operator& op_src, ge::Operator& op_dest) {
        return SUCCESS;
    }
    REGISTER_CUSTOM_OP("DynamicQuantV2")
        .FrameworkType(ONNX)
        .OriginOpType({ge::AscendString("ai.onnx::10::DynamicQuantizeLinear"),
                       ge::AscendString("ai.onnx::11::DynamicQuantizeLinear"),
                       ge::AscendString("ai.onnx::12::DynamicQuantizeLinear"),
                       ge::AscendString("ai.onnx::13::DynamicQuantizeLinear"),
                       ge::AscendString("ai.onnx::14::DynamicQuantizeLinear"),
                       ge::AscendString("ai.onnx::15::DynamicQuantizeLinear")})
        .ParseParamsByOperatorFn(ParseParamsDynamicQuantizeLinear)
        .ImplyType(ImplyType::TVM);
}
