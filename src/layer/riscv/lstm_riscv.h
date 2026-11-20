

#ifndef LAYER_LSTM_RISCV_H
#define LAYER_LSTM_RISCV_H

#include "lstm.h"

namespace ncnn {

class LSTM_riscv : public LSTM
{
public:
    LSTM_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LSTM_RISCV_H
