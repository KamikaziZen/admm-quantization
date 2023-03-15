# Factorize layers: 
```
python scripts/factorize.py --method=admm \
                                    --layer=layer1.0.conv1 \
                                    --reduction-rate=2.0 \
                                    --bits=4 \
                                    --qscheme=tensor_mseminmax_symmetric \
                                    --init=parafac-epc \
                                    --max_iter_als=1000 \
                                    --max_iter_epc=1000 \
                                    --max_iter_admm=1000 \
                                    --model-name=resnet18 \
                                    --with-wandb
```

# Calibrate factorized model:
```
python scripts/calibrate.py --method=admm \
                                    --data-root DATA_ROOT \
                                    --calibration-samples=2048 \
                                    --batch-size=32 \
                                    --bits=4 \
                                    --qscheme=tensor_mseminmax_symmetric \
                                    --reduction-rate=2.0 \
                                    --init=parafac-epc \
                                    --model-name=resnet18 \
                                    --with-wandb
```

# Quantize with custom quantization framework:
```
python scripts/custom_benchmark.py  --model-path=checkpoints/resnet18_m=admm_b=4_r=2.0_i=parafac-epc_s=42_tensor_mseminmax_symmetric_calibrated_2048 \
                                                --model-name=resnet18 \
                                                --data-root DATA_ROOT \
                                                --seed=42 \
                                                --reduction-rate=2.0 \
                                                --batch-size=32 \
                                                --param-bw=4 \
                                                --output-bw=8 \
                                                --bits=4 \
                                                --param-qscheme=tensor_mseminmax_symmetric \
                                                --output-qscheme=histogram \
                                                --observer-samples=500 \
                                                --with-wandb
```

# Quantize with aimet(REQUIRES INSTALLING AIMET FRAMEWORK): 
```
python scripts/quantize_with_aimet.py --method=admm \
                                        --model-name='resnet18' \
                                        --data-root=DATA_ROOT \
                                        --calibration-samples=2048 \
                                        --batch-size=32 \
                                        --bits=4 \
                                        --qscheme=tensor_mseminmax_symmetric \
                                        --reduction-rate=2.0 \
                                        --param_bw=4 \
                                        --output_bw=8 \
                                        --adaround_samples=2048 \
                                        --adaround_iterations=20000 \
                                        --aimet-qscheme=tf_enhanced \
                                        --init=parafac-epc \
                                        --fold \
                                        --with-wandb
```
