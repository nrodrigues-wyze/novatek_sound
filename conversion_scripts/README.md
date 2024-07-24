## convert to useful onnx file
```bash
cd [NOVAIC_TOOL]/release/release/ai_tool/novatek/novaic/ toolchain/closeprefix/bin/compiler/frontend/onnx-onnx

python3 onnx2novaonnx_converter.py \
–-input ../../../../../../test-tutorial/nvtai_tool/input/model/customer/backup2500.onnx
         --output ../../../../../../test-tutorial/nvtai_tool/
  input/model/customer/deploy.onnx
```

## Convert to nvt_model.bin

```bash
cd <PATH_to_novaic>/novatek/novaic/toolchain

./closeprefix/bin/compiler.nvtai \
--config-dir <PATH_to_novaic>/novatek/novaic/test-tutorial/calib_tool \
--pattern-name wyze_sounddetect_16bit --chip 56x
```

