@adb push ../../../checkpoint/translation_encoder.onnx /data/local/tmp/translation_encoder.onnx &&^
adb push ../../../checkpoint/translation_decoder.onnx /data/local/tmp/translation_decoder.onnx &&^
adb logcat -c &&^
cargo apk2 run -p min-trtr-android --release
pause